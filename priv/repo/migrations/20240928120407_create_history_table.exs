defmodule Example.Repo.Migrations.CreateHistoryTable do
  use Ecto.Migration

  def change do
    create table(:titles) do
      add :movies, :text, null: false
      add :embedding, :vector, size: 25

      timestamps()
    end

    create index("titles", ["embedding vector_cosine_ops"], using: :hnsw)
  end
end
