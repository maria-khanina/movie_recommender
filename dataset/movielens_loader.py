"""Download of a MovieLens archive and its import into the project's DuckDB
database.

The archive is streamed to disk, its users, movies and ratings files are copied
out as CSV files, and every CSV file becomes a table of the same name. The import
is done by DuckDB's own CSV reader, so it also works for rating files far larger
than memory. Run as a module: the data directory and the database path come from
the DATA_DIR and DB_PATH environment variables.
"""
import os
import pathlib
import shutil
import zipfile

import duckdb
import requests

from dataset import dataset_settings as settings
from dataset.dataset_settings import DatasetName, TableName


def _download_archive(archive_path: pathlib.Path, dataset_name: DatasetName) -> None:
    archive_url = settings.ARCHIVE_URL_TEMPLATE.format(name=dataset_name)
    with requests.get(
        archive_url,
        stream=True,
        timeout=settings.DOWNLOAD_TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()
        with archive_path.open("wb") as archive_file:
            for chunk in response.iter_content(settings.DOWNLOAD_CHUNK_BYTES):
                archive_file.write(chunk)


def _extract_tables(
    archive_path: pathlib.Path,
    destination_directory: pathlib.Path,
    dataset_name: DatasetName,
) -> None:
    archive_directory = settings.ARCHIVE_DIRECTORY_TEMPLATE.format(name=dataset_name)
    with zipfile.ZipFile(archive_path) as archive:
        for table_name, schema in settings.TABLE_SCHEMAS.items():
            member_name = f"{archive_directory}/{schema['file_name'][dataset_name]}"
            table_path = destination_directory / (
                table_name + settings.RAW_FILE_EXTENSION
            )
            with archive.open(member_name) as source, table_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def download_raw_files(
    destination_directory: pathlib.Path,
    dataset_name: DatasetName,
) -> None:
    """Download a MovieLens archive and keep its tables as CSV files."""
    destination_directory.mkdir(parents=True, exist_ok=True)
    archive_path = destination_directory / f"ml-{dataset_name}.zip"
    _download_archive(archive_path, dataset_name)
    _extract_tables(archive_path, destination_directory, dataset_name)
    archive_path.unlink()


def import_table(
    database_path: str,
    files_directory: pathlib.Path,
    table_name: TableName,
    dataset_name: DatasetName,
) -> None:
    """Create a database table from its CSV file unless the table exists."""
    file_path = files_directory / (table_name + settings.RAW_FILE_EXTENSION)
    schema = settings.TABLE_SCHEMAS[table_name]
    with duckdb.connect(database_path) as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} AS
            SELECT * FROM read_csv(
                '{file_path.as_posix()}',
                delim='{schema['separator'][dataset_name]}',
                header=false,
                names={schema['columns'][dataset_name]},
                encoding='{settings.RAW_FILE_ENCODING}',
                parallel=true
            )
            """
        )


def main() -> None:
    data_directory = pathlib.Path(os.environ[settings.DATA_DIRECTORY_VARIABLE])
    files_directory = data_directory / settings.RAW_FILES_SUBDIRECTORY
    download_raw_files(files_directory, settings.DATASET_NAME)
    for table_name in settings.TABLE_SCHEMAS:
        import_table(
            os.environ[settings.DATABASE_PATH_VARIABLE],
            files_directory,
            table_name,
            settings.DATASET_NAME,
        )


if __name__ == "__main__":
    main()
