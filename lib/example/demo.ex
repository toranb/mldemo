defmodule Example.Demo do
  NimbleCSV.define(DataParser, separator: ",", escape: "\"")

  import Nx.Defn

  defn embed_tokens(embeddings, token_ids) do
    Nx.take(embeddings, token_ids)
  end

  defn normalize(tensor) do
    norm = tensor |> Nx.pow(2) |> Nx.sum() |> Nx.sqrt()
    Nx.divide(tensor, norm)
  end

  defn initialize_embeddings(opts \\ []) do
    vocab_size = opts[:vocab_size]
    key = Nx.Random.key(42)
    Nx.Random.normal_split(key, 0.0, 0.1, shape: {vocab_size, 3}, type: :f16)
  end

  defn pair_loss(embeddings, id_one, id_two) do
    nil
  end

  defn update(embeddings, id_one, id_two, learning_rate) do
    {loss_value, gradients} = value_and_grad(embeddings, fn emb ->
      pair_loss(emb, id_one, id_two)
    end)

    updated_embeddings = Nx.subtract(embeddings, Nx.multiply(learning_rate, gradients))
    {updated_embeddings, loss_value}
  end

  def get_movies() do
    "simple.csv"
    |> File.stream!()
    |> DataParser.parse_stream(skip_headers: false)
    |> Stream.map(fn [one, two] ->
      "#{one} #{two}"
    end)
    |> Enum.to_list()
  end

  def go() do
    examples = get_movies()

    :ok
  end
end
