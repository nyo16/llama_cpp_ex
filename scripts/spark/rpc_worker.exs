# The NIF-hosted RPC worker. Driven by scripts/spark/rpc-worker.sh, which sets
# the environment; run it directly only for debugging.
#
#   SPARK_RPC_ENDPOINT=10.100.64.2:50052 mix run scripts/spark/rpc_worker.exs
#
# Blocks forever on purpose. `LlamaCppEx.RPC.Server` links to this process, and
# upstream's accept loop never returns, so there is nothing to wait on and
# nothing to shut down — the VM is the unit of restart.


# start_link/1 links, and a GenServer whose init/1 returns {:stop, reason}
# exits with that reason — which kills this script before it can say anything
# useful. Trapping turns the exit back into the {:error, reason} that
# start_link is documented to return.
Process.flag(:trap_exit, true)

endpoint =
  System.get_env("SPARK_RPC_ENDPOINT") ||
    raise "SPARK_RPC_ENDPOINT is required, e.g. 10.100.64.2:50052"

cache_dir = System.get_env("SPARK_RPC_CACHE")
n_threads = String.to_integer(System.get_env("SPARK_RPC_THREADS") || "10")

devices =
  case System.get_env("SPARK_RPC_DEVICES") do
    nil -> []
    "" -> []
    list -> String.split(list, ",", trim: true)
  end

case LlamaCppEx.RPC.Server.start_link(
       endpoint: endpoint,
       cache_dir: cache_dir,
       n_threads: n_threads,
       devices: devices
     ) do
  {:ok, pid} ->
    %{devices: served} = LlamaCppEx.RPC.Server.info(pid)
    IO.puts("worker ready: #{endpoint} serving #{Enum.join(served, ", ")}")
    Process.sleep(:infinity)

  {:error, {:rpc, :rpc_unsupported}} ->
    IO.puts(:stderr, """
    This build has no RPC backend. Rebuild the worker with:

        LLAMA_RPC=1 mix compile
    """)

    System.halt(1)

  {:error, reason} ->
    IO.puts(:stderr, "worker failed to start: #{inspect(reason)}")
    System.halt(1)
end
