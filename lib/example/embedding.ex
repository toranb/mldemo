defmodule Example.Embedding do
  import Nx.Defn

  def create_vocabulary(examples) do
    examples
    |> Enum.flat_map(&String.split/1)
    |> Enum.uniq()
    |> Enum.with_index()
    |> Map.new(fn {token, index} -> {token, index} end)
  end

  def preprocess_examples(examples, vocabulary) do
    examples
    |> Enum.flat_map(fn example ->
      movies = String.split(example)
      token_ids = Enum.map(movies, &Map.get(vocabulary, &1, 0))

      token_ids
      |> Enum.with_index()
      |> Enum.flat_map(fn {one, i} ->
        token_ids
        |> Enum.with_index()
        |> Enum.filter(fn {_, j} -> i != j end)
        |> Enum.map(fn {two, _} -> {one, two} end)
      end)
    end)
    |> Enum.uniq()
    |> IO.inspect(label: "examples!")
  end

  def tokenize(text, vocabulary) do
    text
    |> String.split()
    |> Enum.map(&Map.get(vocabulary, &1, 0))
  end

  defn embed_tokens(embeddings, token_ids) do
    Nx.take(embeddings, token_ids)
  end

  defn normalize(tensor) do
    norm =
      tensor
      |> Nx.pow(2)
      |> Nx.sum()
      |> Nx.sqrt()

    Nx.divide(tensor, norm)
  end

  defn pair_loss(embeddings, id_one, id_two) do
    one = embeddings |> embed_tokens(id_one)
    two = embeddings |> embed_tokens(id_two)

    norm_one = normalize(one)
    norm_two = normalize(two)

    similarity = Nx.dot(norm_one, norm_two)
    -similarity
  end

  defn update(embeddings, id_one, id_two, learning_rate) do
    {loss_value, gradients} = value_and_grad(embeddings, fn emb ->
      pair_loss(emb, id_one, id_two)
    end)

    updated_embeddings = Nx.subtract(embeddings, Nx.multiply(learning_rate, gradients))
    {updated_embeddings, loss_value}
  end

  defn initialize_embeddings(opts \\ []) do
    dims = opts[:dims]
    vocab_size = opts[:vocab_size]
    key = Nx.Random.key(42)
    Nx.Random.normal_split(key, 0.0, 0.1, shape: {vocab_size, dims}, type: :f16)
  end

  def train(data, num_epochs, learning_rate) do
    dims = 25
    vocabulary = create_vocabulary(data)
    examples = preprocess_examples(data, vocabulary)
    vocab_size = length(examples)
    embeddings = initialize_embeddings(vocab_size: vocab_size, dims: dims)

    Enum.reduce(1..num_epochs, embeddings, fn epoch, emb ->
      {updated_emb, total_loss} = Enum.reduce(examples, {emb, 0}, fn {one, two}, {current_emb, acc} ->
          {new_emb, loss} = update(current_emb, one, two, learning_rate)
          {new_emb, acc + Nx.to_number(loss)}
        end)

      result = "Epoch #{epoch}, Examples: #{length(examples)}, Loss: #{total_loss / length(examples)}"
      IO.inspect(result)

      updated_emb
    end)
  end

  def get_embedding(movies, vocabulary, embeddings) do
    token_ids = movies
                |> String.split()
                |> Enum.map(&Map.get(vocabulary, &1, 0))
                |> Nx.tensor()

    embeddings
    |> embed_tokens(token_ids)
    |> Nx.mean(axes: [0])
    |> Nx.to_flat_list()
  end
end
