# Usage: LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/chat.exs
#
# Interactive multi-turn chat loop. Type "exit" or "quit" to stop.

model_path =
  System.get_env("LLAMA_MODEL_PATH") ||
    raise "Set LLAMA_MODEL_PATH to a .gguf model file"

:ok = LlamaCppEx.init()
{:ok, model} = LlamaCppEx.load_model(model_path, n_gpu_layers: -1)

system_message = %{role: "system", content: "You are a helpful assistant. Be concise."}

defmodule ChatLoop do
  def run(model, messages) do
    IO.write("\nYou: ")

    case IO.gets("") do
      :eof ->
        IO.puts("\nGoodbye!")

      input ->
        input = String.trim(input)

        if input in ["exit", "quit", ""] do
          IO.puts("Goodbye!")
        else
          messages = messages ++ [%{role: "user", content: input}]

          IO.write("Assistant: ")

          chunks =
            model
            |> LlamaCppEx.stream_chat(messages, max_tokens: 512, temp: 0.7)
            |> Enum.map(fn chunk ->
              IO.write(chunk)
              chunk
            end)

          reply = Enum.join(chunks)
          IO.puts("")

          messages = messages ++ [%{role: "assistant", content: reply}]
          run(model, messages)
        end
    end
  end
end

IO.puts("Chat with #{LlamaCppEx.Model.desc(model)}")
IO.puts("Type 'exit' or 'quit' to stop.\n")

ChatLoop.run(model, [system_message])
