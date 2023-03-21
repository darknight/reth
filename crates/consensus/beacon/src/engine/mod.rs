use futures::{Future, FutureExt, StreamExt};
use reth_db::{database::Database, tables, transaction::DbTx};
use reth_executor::blockchain_tree::{BlockStatus, BlockchainTree};
use reth_interfaces::{
    consensus::{Consensus, ForkchoiceState},
    executor::Error as ExecutorError,
    sync::SyncStateUpdater,
    Error,
};
use reth_primitives::{BlockHash, SealedBlock, H256};
use reth_provider::ExecutorFactory;
use reth_rpc_types::engine::{
    ExecutionPayload, ForkchoiceUpdated, PayloadAttributes, PayloadStatus, PayloadStatusEnum,
};
use reth_stages::Pipeline;
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::*;

mod error;
pub use error::{BeaconEngineError, BeaconEngineResult};

mod message;
pub use message::{BeaconEngineMessage, BeaconEngineSender};

mod pipeline_state;
pub use pipeline_state::PipelineState;

#[derive(Debug, Default)]
enum BeaconEngineAction {
    #[default]
    None,
    RunPipeline,
}

impl BeaconEngineAction {
    fn run_pipeline(&self) -> bool {
        matches!(self, BeaconEngineAction::RunPipeline)
    }
}

/// The beacon consensus engine is the driver that switches between historical and live sync.
///
/// TODO: add more docs
#[must_use = "Future does nothing unless polled"]
pub struct BeaconConsensusEngine<
    DB: Database,
    U: SyncStateUpdater,
    C: Consensus,
    EF: ExecutorFactory,
> {
    db: Arc<DB>,
    pipeline_state: Option<PipelineState<DB, U>>,
    blockchain_tree: BlockchainTree<DB, C, EF>,
    message_rx: UnboundedReceiverStream<BeaconEngineMessage>,
    forkchoice_state: Option<ForkchoiceState>,
    next_action: BeaconEngineAction,
}

impl<DB, U, C, EF> BeaconConsensusEngine<DB, U, C, EF>
where
    DB: Database + Unpin + 'static,
    U: SyncStateUpdater + Unpin + 'static,
    C: Consensus,
    EF: ExecutorFactory + 'static,
{
    /// Create new instance of the [BeaconConsensusEngine].
    pub fn new(
        db: Arc<DB>,
        pipeline: Pipeline<DB, U>,
        blockchain_tree: BlockchainTree<DB, C, EF>,
        message_rx: UnboundedReceiver<BeaconEngineMessage>,
    ) -> Self {
        Self {
            db,
            pipeline_state: Some(PipelineState::Idle(pipeline)),
            blockchain_tree,
            message_rx: UnboundedReceiverStream::new(message_rx),
            forkchoice_state: None,
            next_action: BeaconEngineAction::RunPipeline,
        }
    }

    /// Returns `true` if the pipeline is currently idle.
    fn pipeline_is_idle(&self) -> bool {
        self.pipeline_state.as_ref().expect("pipeline state is set").is_idle()
    }

    /// Set next action to [BeaconEngineAction::RunPipeline] to indicate that
    /// consensus engine needs to run the pipeline as soon as it becomes available.
    fn pipeline_run_needed(&mut self) {
        self.next_action = BeaconEngineAction::RunPipeline;
    }

    /// Called to resolve chain forks and ensure that the Execution layer is working with the latest
    /// valid chain.
    ///
    /// These responses should adhere to the [Engine API Spec for
    /// `engine_forkchoiceUpdated`](https://github.com/ethereum/execution-apis/blob/main/src/engine/paris.md#specification-1).
    fn on_forkchoice_updated(
        &mut self,
        state: ForkchoiceState,
        _attrs: Option<PayloadAttributes>,
    ) -> ForkchoiceUpdated {
        trace!(target: "consensus::engine", ?state, "Received new forkchoice state");
        if state.head_block_hash.is_zero() {
            return ForkchoiceUpdated::new(PayloadStatus::from_status(PayloadStatusEnum::Invalid {
                validation_error: BeaconEngineError::ForkchoiceEmptyHead.to_string(),
            }))
        }

        self.forkchoice_state = Some(state.clone());
        let status = if self.pipeline_is_idle() {
            match self.blockchain_tree.make_canonical(&state.head_block_hash) {
                Ok(_) => PayloadStatus::from_status(PayloadStatusEnum::Valid),
                Err(error) => {
                    error!(target: "consensus::engine", ?state, ?error, "Error canonicalizing the head hash");
                    self.pipeline_run_needed();
                    match error {
                        Error::Execution(ExecutorError::BlockPreMerge) => {
                            PayloadStatus::from_status(PayloadStatusEnum::Invalid {
                                validation_error: ExecutorError::BlockPreMerge.to_string(),
                            })
                            .with_latest_valid_hash(H256::zero())
                        }
                        _ => PayloadStatus::from_status(PayloadStatusEnum::Syncing),
                    }
                }
            }
        } else {
            PayloadStatus::from_status(PayloadStatusEnum::Syncing)
        };
        ForkchoiceUpdated::new(status)
    }

    /// When the Consensus layer receives a new block via the consensus gossip protocol,
    /// the transactions in the block are sent to the execution layer in the form of a
    /// `ExecutionPayload`. The Execution layer executes the transactions and validates the
    /// state in the block header, then passes validation data back to Consensus layer, that
    /// adds the block to the head of its own blockchain and attests to it. The block is then
    /// broadcasted over the consensus p2p network in the form of a "Beacon block".
    ///
    /// These responses should adhere to the [Engine API Spec for
    /// `engine_newPayload`](https://github.com/ethereum/execution-apis/blob/main/src/engine/paris.md#specification).
    fn on_new_payload(&mut self, payload: ExecutionPayload) -> PayloadStatus {
        let block = match SealedBlock::try_from(payload) {
            Ok(block) => block,
            Err(error) => {
                return PayloadStatus::from_status(PayloadStatusEnum::InvalidBlockHash {
                    validation_error: error.to_string(),
                })
            }
        };

        if self.pipeline_is_idle() {
            let block_hash = block.hash;
            match self.blockchain_tree.insert_block(block) {
                Ok(status) => {
                    let latest_valid_hash =
                        matches!(status, BlockStatus::Valid).then_some(block_hash);
                    let status = match status {
                        BlockStatus::Valid => PayloadStatusEnum::Valid,
                        BlockStatus::Accepted => PayloadStatusEnum::Accepted,
                        BlockStatus::Disconnected => PayloadStatusEnum::Syncing,
                    };
                    PayloadStatus::new(status, latest_valid_hash)
                }
                Err(error) => {
                    let latest_valid_hash =
                        matches!(error, Error::Execution(ExecutorError::BlockPreMerge))
                            .then_some(H256::zero());
                    PayloadStatus::new(
                        PayloadStatusEnum::Invalid { validation_error: error.to_string() },
                        latest_valid_hash,
                    )
                }
            }
        } else {
            PayloadStatus::from_status(PayloadStatusEnum::Syncing)
        }
    }

    /// Returns the next pipeline state depending on the current value of the next action.
    /// Resets the next action to the default value.
    fn next_pipeline_state(
        &mut self,
        pipeline: Pipeline<DB, U>,
        tip: H256,
    ) -> PipelineState<DB, U> {
        let next_action = std::mem::take(&mut self.next_action);
        if next_action.run_pipeline() {
            trace!(target: "consensus::engine", ?tip, "Starting the pipeline");
            PipelineState::Running(pipeline.run_as_fut(self.db.clone(), tip))
        } else {
            PipelineState::Idle(pipeline)
        }
    }

    /// Attempt to restore the tree with the finalized block number.
    /// If the finalized block is missing from the database, trigger the pipeline run.
    fn restore_tree_if_possible(&mut self, finalized_hash: BlockHash) -> BeaconEngineResult<()> {
        match self.db.view(|tx| tx.get::<tables::HeaderNumbers>(finalized_hash))?? {
            Some(number) => self.blockchain_tree.restore_canonical_hashes(number)?,
            None => self.pipeline_run_needed(),
        };
        Ok(())
    }
}

/// On initialization, the consensus engine will poll the message receiver and return
/// [Poll::Pending] until the first forkchoice update message is received.
///
/// As soon as the consensus engine receives the first forkchoice updated message and updates the
/// local forkchoice state, it will launch the pipeline to sync to the head hash.
/// While the pipeline is syncing, the consensus engine will keep processing messages from the
/// receiver and forwarding them to the blockchain tree.
impl<DB, U, C, EF> Future for BeaconConsensusEngine<DB, U, C, EF>
where
    DB: Database + Unpin + 'static,
    U: SyncStateUpdater + Unpin + 'static,
    C: Consensus + Unpin,
    EF: ExecutorFactory + Unpin + 'static,
{
    type Output = Result<(), BeaconEngineError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        // Set the next pipeline state.
        loop {
            // Process all incoming messages first.
            while let Poll::Ready(Some(msg)) = this.message_rx.poll_next_unpin(cx) {
                match msg {
                    BeaconEngineMessage::ForkchoiceUpdated(state, attrs, tx) => {
                        let response = this.on_forkchoice_updated(state, attrs);
                        let _ = tx.send(Ok(response));
                    }
                    BeaconEngineMessage::NewPayload(block, tx) => {
                        let response = this.on_new_payload(block);
                        let _ = tx.send(Ok(response));
                    }
                }
            }

            // Lookup the forkchoice state. We can't launch the pipeline without the tip.
            let forckchoice_state = match &this.forkchoice_state {
                Some(state) => state,
                None => return Poll::Pending,
            };

            let tip = forckchoice_state.head_block_hash;
            let next_state = match this.pipeline_state.take().expect("pipeline state is set") {
                PipelineState::Running(mut fut) => {
                    match fut.poll_unpin(cx) {
                        Poll::Ready((pipeline, result)) => {
                            // Any pipeline error at this point is fatal.
                            if let Err(error) = result {
                                return Poll::Ready(Err(error.into()))
                            }

                            // Update the state and hashes of the blockchain tree if possible
                            if let Err(error) = this
                                .restore_tree_if_possible(forckchoice_state.finalized_block_hash)
                            {
                                error!(target: "consensus::engine", ?error, "Error restoring blockchain tree");
                                return Poll::Ready(Err(error.into()))
                            }

                            // Get next pipeline state.
                            this.next_pipeline_state(pipeline, tip)
                        }
                        Poll::Pending => {
                            this.pipeline_state = Some(PipelineState::Running(fut));
                            return Poll::Pending
                        }
                    }
                }
                PipelineState::Idle(pipeline) => this.next_pipeline_state(pipeline, tip),
            };
            this.pipeline_state = Some(next_state);

            // If the pipeline is idle, break from the loop.
            if this.pipeline_is_idle() {
                return Poll::Pending
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use reth_db::mdbx::{test_utils::create_test_rw_db, Env, WriteMap};
    use reth_executor::{
        blockchain_tree::{config::BlockchainTreeConfig, externals::TreeExternals},
        post_state::PostState,
        test_utils::TestExecutorFactory,
    };
    use reth_interfaces::{sync::NoopSyncStateUpdate, test_utils::TestConsensus};
    use reth_primitives::{ChainSpec, ChainSpecBuilder, SealedBlockWithSenders, H256, MAINNET};
    use reth_provider::Transaction;
    use reth_stages::{test_utils::TestStages, ExecOutput, PipelineError, StageError};
    use std::{collections::VecDeque, time::Duration};
    use tokio::sync::{
        mpsc::{unbounded_channel, UnboundedSender},
        oneshot::{self, error::TryRecvError},
        watch,
    };

    type TestBeaconConsensusEngine = BeaconConsensusEngine<
        Env<WriteMap>,
        NoopSyncStateUpdate,
        TestConsensus,
        TestExecutorFactory,
    >;

    struct TestEnv<DB> {
        db: Arc<DB>,
        // Keep the tip receiver around, so it's not dropped.
        #[allow(dead_code)]
        tip_rx: watch::Receiver<H256>,
        sync_tx: UnboundedSender<BeaconEngineMessage>,
    }

    impl<DB> TestEnv<DB> {
        fn new(
            db: Arc<DB>,
            tip_rx: watch::Receiver<H256>,
            sync_tx: UnboundedSender<BeaconEngineMessage>,
        ) -> Self {
            Self { db, tip_rx, sync_tx }
        }

        fn send_new_payload(
            &self,
            block: ExecutionPayload,
        ) -> oneshot::Receiver<BeaconEngineResult<PayloadStatus>> {
            let (tx, rx) = oneshot::channel();
            self.sync_tx
                .send(BeaconEngineMessage::NewPayload(block, tx))
                .expect("failed to send msg");
            rx
        }

        fn send_forkchoice_updated(
            &self,
            state: ForkchoiceState,
        ) -> oneshot::Receiver<BeaconEngineResult<ForkchoiceUpdated>> {
            let (tx, rx) = oneshot::channel();
            self.sync_tx
                .send(BeaconEngineMessage::ForkchoiceUpdated(state, None, tx))
                .expect("failed to send msg");
            rx
        }
    }

    fn setup_consensus_engine(
        chain_spec: Arc<ChainSpec>,
        pipeline_exec_outputs: VecDeque<Result<ExecOutput, StageError>>,
        executor_results: Vec<PostState>,
    ) -> (TestBeaconConsensusEngine, TestEnv<Env<WriteMap>>) {
        reth_tracing::init_test_tracing();
        let db = create_test_rw_db();
        let consensus = TestConsensus::default();
        let executor_factory = TestExecutorFactory::new(chain_spec.clone());
        executor_factory.extend(executor_results);

        // Setup pipeline
        let (tip_tx, tip_rx) = watch::channel(H256::default());
        let pipeline = Pipeline::builder()
            .add_stages(TestStages::new(pipeline_exec_outputs, Default::default()))
            .with_tip_sender(tip_tx)
            .build();

        // Setup blockchain tree
        let externals =
            TreeExternals::new(db.clone(), consensus, executor_factory, chain_spec.clone());
        let config = BlockchainTreeConfig::new(1, 2, 3);
        let tree = BlockchainTree::new(externals, config).expect("failed to create tree");

        let (sync_tx, sync_rx) = unbounded_channel();
        (
            BeaconConsensusEngine::new(db.clone(), pipeline, tree, sync_rx),
            TestEnv::new(db, tip_rx, sync_tx),
        )
    }

    fn spawn_consensus_engine(
        engine: TestBeaconConsensusEngine,
    ) -> oneshot::Receiver<Result<(), BeaconEngineError>> {
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let result = engine.await;
            tx.send(result).expect("failed to forward consensus engine result");
        });
        rx
    }

    // Pipeline error is propagated.
    #[tokio::test]
    async fn pipeline_error_is_propagated() {
        let chain_spec = Arc::new(
            ChainSpecBuilder::default()
                .chain(MAINNET.chain)
                .genesis(MAINNET.genesis.clone())
                .paris_activated()
                .build(),
        );
        let (consensus_engine, env) = setup_consensus_engine(
            chain_spec,
            VecDeque::from([Err(StageError::ChannelClosed)]),
            Vec::default(),
        );
        let rx = spawn_consensus_engine(consensus_engine);

        let _ = env.send_forkchoice_updated(ForkchoiceState {
            head_block_hash: H256::random(),
            ..Default::default()
        });
        assert_matches!(
            rx.await,
            Ok(Err(BeaconEngineError::Pipeline(PipelineError::Stage(StageError::ChannelClosed))))
        );
    }

    // Test that the consensus engine is idle until first forkchoice updated is received.
    #[tokio::test]
    async fn is_idle_until_forkchoice_is_set() {
        let chain_spec = Arc::new(
            ChainSpecBuilder::default()
                .chain(MAINNET.chain)
                .genesis(MAINNET.genesis.clone())
                .paris_activated()
                .build(),
        );
        let (consensus_engine, env) = setup_consensus_engine(
            chain_spec,
            VecDeque::from([Err(StageError::ChannelClosed)]),
            Vec::default(),
        );
        let mut rx = spawn_consensus_engine(consensus_engine);

        // consensus engine is idle
        std::thread::sleep(Duration::from_millis(100));
        assert_matches!(rx.try_recv(), Err(TryRecvError::Empty));

        // consensus engine is still idle
        let _ = env.send_new_payload(SealedBlock::default().into());
        assert_matches!(rx.try_recv(), Err(TryRecvError::Empty));

        // consensus engine receives a forkchoice state and triggers the pipeline
        let _ = env.send_forkchoice_updated(ForkchoiceState {
            head_block_hash: H256::random(),
            ..Default::default()
        });
        assert_matches!(
            rx.await,
            Ok(Err(BeaconEngineError::Pipeline(PipelineError::Stage(StageError::ChannelClosed))))
        );
    }

    // Test that the consensus engine runs the pipeline again if the tree cannot be restored.
    // The consensus engine will propagate the second result (error) only if it runs the pipeline
    // for the second time.
    #[tokio::test]
    async fn runs_pipeline_again_if_tree_not_restored() {
        let chain_spec = Arc::new(
            ChainSpecBuilder::default()
                .chain(MAINNET.chain)
                .genesis(MAINNET.genesis.clone())
                .paris_activated()
                .build(),
        );
        let (consensus_engine, env) = setup_consensus_engine(
            chain_spec,
            VecDeque::from([
                Ok(ExecOutput { stage_progress: 1, done: true }),
                Err(StageError::ChannelClosed),
            ]),
            Vec::default(),
        );
        let rx = spawn_consensus_engine(consensus_engine);

        let _ = env.send_forkchoice_updated(ForkchoiceState {
            head_block_hash: H256::random(),
            ..Default::default()
        });
        assert_matches!(
            rx.await,
            Ok(Err(BeaconEngineError::Pipeline(PipelineError::Stage(StageError::ChannelClosed))))
        );
    }

    fn insert_blocks<'a, DB: Database>(db: &DB, mut blocks: impl Iterator<Item = &'a SealedBlock>) {
        let mut transaction = Transaction::new(db).unwrap();
        blocks
            .try_for_each(|b| {
                transaction
                    .insert_block(SealedBlockWithSenders::new(b.clone(), Vec::default()).unwrap())
            })
            .expect("failed to insert");
        transaction.commit().unwrap();
    }

    mod fork_choice_updated {
        use super::*;
        use reth_interfaces::test_utils::generators::random_block;

        #[tokio::test]
        async fn empty_head() {
            let chain_spec = Arc::new(
                ChainSpecBuilder::default()
                    .chain(MAINNET.chain)
                    .genesis(MAINNET.genesis.clone())
                    .paris_activated()
                    .build(),
            );
            let (consensus_engine, env) = setup_consensus_engine(
                chain_spec,
                VecDeque::from([Ok(ExecOutput { done: true, stage_progress: 0 })]),
                Vec::default(),
            );

            let mut engine_rx = spawn_consensus_engine(consensus_engine);

            let rx = env.send_forkchoice_updated(ForkchoiceState::default());
            let expected_result = ForkchoiceUpdated::from_status(PayloadStatusEnum::Invalid {
                validation_error: BeaconEngineError::ForkchoiceEmptyHead.to_string(),
            });
            assert_matches!(rx.await, Ok(Ok(result)) => assert_eq!(result, expected_result));

            assert_matches!(engine_rx.try_recv(), Err(TryRecvError::Empty));
        }

        #[tokio::test]
        async fn unknown_head_hash() {
            let chain_spec = Arc::new(
                ChainSpecBuilder::default()
                    .chain(MAINNET.chain)
                    .genesis(MAINNET.genesis.clone())
                    .paris_activated()
                    .build(),
            );
            let (consensus_engine, env) = setup_consensus_engine(
                chain_spec,
                VecDeque::from([Ok(ExecOutput { done: true, stage_progress: 0 })]),
                Vec::default(),
            );

            let genesis = random_block(0, None, None, Some(0));
            let block1 = random_block(1, Some(genesis.hash), None, Some(0));
            insert_blocks(env.db.as_ref(), [&genesis, &block1].into_iter());

            let mut engine_rx = spawn_consensus_engine(consensus_engine);

            let rx_invalid = env.send_forkchoice_updated(ForkchoiceState {
                head_block_hash: H256::random(),
                finalized_block_hash: block1.hash,
                ..Default::default()
            });
            let expected_result = ForkchoiceUpdated::from_status(PayloadStatusEnum::Syncing);
            assert_matches!(rx_invalid.await, Ok(Ok(result)) => assert_eq!(result, expected_result));

            let rx_valid = env.send_forkchoice_updated(ForkchoiceState {
                head_block_hash: block1.hash,
                finalized_block_hash: block1.hash,
                ..Default::default()
            });
            let expected_result = ForkchoiceUpdated::from_status(PayloadStatusEnum::Valid);
            assert_matches!(rx_valid.await, Ok(Ok(result)) => assert_eq!(result, expected_result));

            assert_matches!(engine_rx.try_recv(), Err(TryRecvError::Empty));
        }

        #[tokio::test]
        async fn unknown_finalized_hash() {
            let chain_spec = Arc::new(
                ChainSpecBuilder::default()
                    .chain(MAINNET.chain)
                    .genesis(MAINNET.genesis.clone())
                    .paris_activated()
                    .build(),
            );
            let (consensus_engine, env) = setup_consensus_engine(
                chain_spec,
                VecDeque::from([Ok(ExecOutput { done: true, stage_progress: 0 })]),
                Vec::default(),
            );

            let genesis = random_block(0, None, None, Some(0));
            let block1 = random_block(1, Some(genesis.hash), None, Some(0));
            insert_blocks(env.db.as_ref(), [&genesis, &block1].into_iter());

            let _ = spawn_consensus_engine(consensus_engine);

            let rx = env.send_forkchoice_updated(ForkchoiceState {
                head_block_hash: H256::random(),
                finalized_block_hash: block1.hash,
                ..Default::default()
            });
            let expected_result = ForkchoiceUpdated::from_status(PayloadStatusEnum::Syncing);
            assert_matches!(rx.await, Ok(Ok(result)) => assert_eq!(result, expected_result));
        }

        #[tokio::test]
        async fn forkchoice_updated_invalid_pow() {
            let chain_spec = Arc::new(
                ChainSpecBuilder::default()
                    .chain(MAINNET.chain)
                    .genesis(MAINNET.genesis.clone())
                    .london_activated()
                    .build(),
            );
            let (consensus_engine, env) = setup_consensus_engine(
                chain_spec,
                VecDeque::from([
                    Ok(ExecOutput { done: true, stage_progress: 0 }),
                    Ok(ExecOutput { done: true, stage_progress: 0 }),
                ]),
                Vec::default(),
            );

            let genesis = random_block(0, None, None, Some(0));
            let block1 = random_block(1, Some(genesis.hash), None, Some(0));

            insert_blocks(env.db.as_ref(), [&genesis, &block1].into_iter());

            let _ = spawn_consensus_engine(consensus_engine);

            let rx = env.send_forkchoice_updated(ForkchoiceState {
                head_block_hash: H256::random(),
                finalized_block_hash: block1.hash,
                ..Default::default()
            });
            let expected_result = ForkchoiceUpdated::from_status(PayloadStatusEnum::Syncing);
            assert_matches!(rx.await, Ok(Ok(result)) => assert_eq!(result, expected_result));

            let rx = env.send_forkchoice_updated(ForkchoiceState {
                head_block_hash: block1.hash,
                finalized_block_hash: block1.hash,
                ..Default::default()
            });
            let expected_result = ForkchoiceUpdated::from_status(PayloadStatusEnum::Invalid {
                validation_error: ExecutorError::BlockPreMerge.to_string(),
            })
            .with_latest_valid_hash(H256::zero());
            assert_matches!(rx.await, Ok(Ok(result)) => assert_eq!(result, expected_result));
        }
    }

    mod new_payload {
        use super::*;
        use reth_interfaces::{
            executor::Error as ExecutorError, test_utils::generators::random_block,
        };
        use reth_primitives::{Hardfork, U256};
        use reth_provider::test_utils::blocks::BlockChainTestData;

        #[tokio::test]
        async fn payload_known() {
            let chain_spec = Arc::new(
                ChainSpecBuilder::default()
                    .chain(MAINNET.chain)
                    .genesis(MAINNET.genesis.clone())
                    .paris_activated()
                    .build(),
            );
            let (consensus_engine, env) = setup_consensus_engine(
                chain_spec,
                VecDeque::from([Ok(ExecOutput { done: true, stage_progress: 0 })]),
                Vec::default(),
            );

            let genesis = random_block(0, None, None, Some(0));
            let block1 = random_block(1, Some(genesis.hash), None, Some(0));
            let block2 = random_block(2, Some(block1.hash), None, Some(0));
            insert_blocks(env.db.as_ref(), [&genesis, &block1, &block2].into_iter());

            let mut engine_rx = spawn_consensus_engine(consensus_engine);

            // Send forkchoice
            let rx = env.send_forkchoice_updated(ForkchoiceState {
                head_block_hash: block1.hash,
                finalized_block_hash: block1.hash,
                ..Default::default()
            });
            let expected_result =
                ForkchoiceUpdated::new(PayloadStatus::from_status(PayloadStatusEnum::Syncing));
            assert_matches!(rx.await, Ok(Ok(result)) => assert_eq!(result, expected_result));

            // Send new payload
            let rx = env.send_new_payload(block2.clone().into());
            let expected_result = PayloadStatus::from_status(PayloadStatusEnum::Valid)
                .with_latest_valid_hash(block2.hash);
            assert_matches!(rx.await, Ok(Ok(result)) => assert_eq!(result, expected_result));

            assert_matches!(engine_rx.try_recv(), Err(TryRecvError::Empty));
        }

        #[tokio::test]
        async fn payload_parent_unknown() {
            let chain_spec = Arc::new(
                ChainSpecBuilder::default()
                    .chain(MAINNET.chain)
                    .genesis(MAINNET.genesis.clone())
                    .paris_activated()
                    .build(),
            );
            let (consensus_engine, env) = setup_consensus_engine(
                chain_spec,
                VecDeque::from([Ok(ExecOutput { done: true, stage_progress: 0 })]),
                Vec::default(),
            );

            let genesis = random_block(0, None, None, Some(0));

            insert_blocks(env.db.as_ref(), [&genesis].into_iter());

            let mut engine_rx = spawn_consensus_engine(consensus_engine);

            // Send forkchoice
            let rx = env.send_forkchoice_updated(ForkchoiceState {
                head_block_hash: genesis.hash,
                finalized_block_hash: genesis.hash,
                ..Default::default()
            });
            let expected_result =
                ForkchoiceUpdated::new(PayloadStatus::from_status(PayloadStatusEnum::Syncing));
            assert_matches!(rx.await, Ok(Ok(result)) => assert_eq!(result, expected_result));

            // Send new payload
            let block = random_block(2, Some(H256::random()), None, Some(0));
            let rx = env.send_new_payload(block.into());
            let expected_result = PayloadStatus::from_status(PayloadStatusEnum::Syncing);
            assert_matches!(rx.await, Ok(Ok(result)) => assert_eq!(result, expected_result));

            assert_matches!(engine_rx.try_recv(), Err(TryRecvError::Empty));
        }

        #[tokio::test]
        async fn payload_pre_merge() {
            let data = BlockChainTestData::default();
            let mut block1 = data.blocks[0].0.block.clone();
            block1.header.difficulty = MAINNET.fork(Hardfork::Paris).ttd().unwrap() - U256::from(1);
            block1 = block1.unseal().seal_slow();
            let (block2, exec_result2) = data.blocks[1].clone();
            let mut block2 = block2.block;
            block2.withdrawals = None;
            block2.header.parent_hash = block1.hash;
            block2.header.base_fee_per_gas = Some(100);
            block2.header.difficulty = U256::ZERO;
            block2 = block2.unseal().seal_slow();

            let chain_spec = Arc::new(
                ChainSpecBuilder::default()
                    .chain(MAINNET.chain)
                    .genesis(MAINNET.genesis.clone())
                    .london_activated()
                    .build(),
            );
            let (consensus_engine, env) = setup_consensus_engine(
                chain_spec,
                VecDeque::from([Ok(ExecOutput { done: true, stage_progress: 0 })]),
                Vec::from([exec_result2]),
            );

            insert_blocks(env.db.as_ref(), [&data.genesis, &block1].into_iter());

            let mut engine_rx = spawn_consensus_engine(consensus_engine);

            // Send forkchoice
            let rx = env.send_forkchoice_updated(ForkchoiceState {
                head_block_hash: block1.hash,
                finalized_block_hash: block1.hash,
                ..Default::default()
            });
            let expected_result =
                ForkchoiceUpdated::new(PayloadStatus::from_status(PayloadStatusEnum::Syncing));
            assert_matches!(rx.await, Ok(Ok(result)) => assert_eq!(result, expected_result));

            // Send new payload
            let rx = env.send_new_payload(block2.into());
            let expected_result = PayloadStatus::from_status(PayloadStatusEnum::Invalid {
                validation_error: ExecutorError::BlockPreMerge.to_string(),
            })
            .with_latest_valid_hash(H256::zero());
            assert_matches!(rx.await, Ok(Ok(result)) => assert_eq!(result, expected_result));

            assert_matches!(engine_rx.try_recv(), Err(TryRecvError::Empty));
        }
    }
}