-- CardioSpark MySQL 元数据层初始化脚本。
-- 运行前先创建数据库：CREATE DATABASE cardiospark CHARACTER SET utf8mb4;
USE cardiospark;

CREATE TABLE IF NOT EXISTS upload_tasks (
    task_id VARCHAR(128) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_size BIGINT NOT NULL DEFAULT 0,
    total_chunks INT NOT NULL DEFAULT 0,
    uploaded_chunks INT NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL,
    storage_mode VARCHAR(20) NOT NULL DEFAULT 'local',
    storage_path TEXT,
    dataset_id VARCHAR(128),
    data_period VARCHAR(32),
    result_path TEXT,
    stage VARCHAR(32),
    progress DOUBLE NOT NULL DEFAULT 0,
    started_at VARCHAR(64),
    finished_at VARCHAR(64),
    file_sha256 VARCHAR(128),
    error_message TEXT,
    created_at VARCHAR(64) NOT NULL,
    updated_at VARCHAR(64) NOT NULL,
    INDEX idx_upload_status_updated (status, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS upload_chunks (
    task_id VARCHAR(128) NOT NULL,
    chunk_index INT NOT NULL,
    chunk_size BIGINT NOT NULL,
    chunk_path TEXT NOT NULL,
    created_at VARCHAR(64) NOT NULL,
    PRIMARY KEY (task_id, chunk_index),
    CONSTRAINT fk_chunk_task FOREIGN KEY (task_id)
        REFERENCES upload_tasks(task_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id VARCHAR(128) PRIMARY KEY,
    task_id VARCHAR(128) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_size BIGINT NOT NULL DEFAULT 0,
    file_sha256 VARCHAR(128),
    storage_mode VARCHAR(20) NOT NULL DEFAULT 'local',
    hdfs_path TEXT,
    local_path TEXT,
    data_period VARCHAR(32),
    region_level VARCHAR(32),
    region_name VARCHAR(255),
    row_count BIGINT,
    validation_status VARCHAR(32) NOT NULL DEFAULT 'pending',
    result_path TEXT,
    upload_time VARCHAR(64) NOT NULL,
    INDEX idx_dataset_period (data_period),
    CONSTRAINT fk_dataset_task FOREIGN KEY (task_id)
        REFERENCES upload_tasks(task_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS assessment_records (
    assessment_id VARCHAR(128) PRIMARY KEY,
    client_hash VARCHAR(128) NOT NULL,
    model_version VARCHAR(128),
    knowledge_version VARCHAR(128),
    heart_probability DOUBLE NOT NULL,
    stroke_probability DOUBLE NOT NULL,
    risk_level_code INT NOT NULL,
    risk_level_name VARCHAR(64) NOT NULL,
    final_category INT NOT NULL,
    input_json LONGTEXT NOT NULL,
    result_json LONGTEXT NOT NULL,
    created_at VARCHAR(64) NOT NULL,
    INDEX idx_assessment_client_created (client_hash, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
