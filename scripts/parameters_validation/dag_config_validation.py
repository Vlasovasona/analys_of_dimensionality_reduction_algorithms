def validate_storage_config(
    bucket_name: str,
    processed_prefix: str,
    local_data_dir: str,
) -> None:
    """
    Валидирует параметры хранения и загрузки данных.
    """

    if not bucket_name or not bucket_name.strip():
        raise ValueError("bucket_name не может быть пустым")

    if not processed_prefix or not processed_prefix.strip():
        raise ValueError("processed_prefix не может быть пустым")

    if not local_data_dir or not local_data_dir.strip():
        raise ValueError("local_data_dir не может быть пустым")

    return None