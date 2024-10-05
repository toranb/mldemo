defmodule Example.Recommendation do
  NimbleCSV.define(ZataParser, separator: ",", escape: "\"")

  def get_movies() do
    "pairs.csv"
    |> File.stream!()
    |> ZataParser.parse_stream()
    |> Stream.map(fn [one, two] ->
      "#{one} #{two}"
    end)
    |> Enum.to_list()
  end

  def train() do
    examples = get_movies()
    vocabulary = Example.Embedding.create_vocabulary(examples)

    num_epochs = 20
    learning_rate = 0.01

    model = Example.Embedding.train(examples, num_epochs, learning_rate)
    serialized_container = Nx.serialize(model)
    File.write!("#{Path.dirname(__ENV__.file)}/model_data", serialized_container)

    "pairs.csv"
    |> File.stream!()
    |> ZataParser.parse_stream()
    |> Stream.map(fn [one, two] ->
      movies = "#{one} #{two}"
      result = Example.Embedding.get_embedding(movies, vocabulary, model) |> Nx.tensor()
      result = result |> Example.Embedding.normalize()

      %Example.Title{}
      |> Example.Title.changeset(%{
        movies: movies,
        embedding: result
      })
    end)
    |> Enum.each(fn embedding ->
      embedding |> Example.Repo.insert!()
    end)
  end

  def guess(movies) do
    examples = get_movies()

    model_data = File.read!("#{Path.dirname(__ENV__.file)}/model_data")
    model = Nx.deserialize(model_data)

    vocabulary = Example.Embedding.create_vocabulary(examples)

    result = Example.Embedding.get_embedding(movies, vocabulary, model) |> Nx.tensor()
    result = result |> Example.Embedding.normalize()
    results = Example.Title.search(result)

    results
    |> Enum.each(fn title ->
      vector = title.embedding |> Pgvector.to_tensor()

      norm_one = Example.Embedding.normalize(vector)
      norm_two = Example.Embedding.normalize(result)
      similarity = Nx.dot(norm_one, norm_two)

      similarity |> IO.inspect(label: "similarity")
      title.movies |> IO.inspect(label: "movies")
    end)

    results
  end
end
