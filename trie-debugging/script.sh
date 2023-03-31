set -ex
DB=./sepolia-db
# run this on the backup database's copy

run() {
    DBG_DB=$DB-$1
    # rm -rf $DBG_DB
    # echo "running to $1, copying to $DBG_DB"
    # cp -rf $DB $DBG_DB
    # ./reth drop-stage --db $DBG_DB execution --chain=sepolia
    # ./reth drop-stage --db $DBG_DB hashed-accounts
    # ./reth drop-stage --db $DBG_DB hashed-storage
    ./reth drop-stage --db ./sepolia-db-2320000-master merkle

    # RUST_LOG=info ../../reth-master/target/debug-fast/reth node \
    #     --chain=sepolia \
    #     --debug.tip 0x696d95da6726a67afd5be2a37d3883e9be8008491b30d5bd1069ea5922fa2a41 \
    #     --db $DBG_DB --debug.max-block 2320001
    # RUST_LOG=info cargo r --profile=debug-fast --bin reth -- node \
    #     --chain=sepolia \
    #     --debug.tip 0x696d95da6726a67afd5be2a37d3883e9be8008491b30d5bd1069ea5922fa2a41 \
    #     --db $DBG_DB --debug.max-block 2320001
    #     # 2325000 - bad
    #     # 2313000 - good
    #     # 2312500 - good
}

run 5000


####
# 1. Clear execution
# 2. force rehashing & remerklziing of everything every time
# 3. run to next block
