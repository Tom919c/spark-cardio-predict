"""根据配置创建 SQLite 或 MySQL 元数据库 DAO。"""

from src.dao.database_dao import DatabaseDAO


def create_database_dao(config):
    database_type = str(config.get("DATABASE_TYPE", "sqlite")).lower()
    if database_type == "mysql":
        from src.dao.mysql_dao import MySQLDatabaseDAO

        return MySQLDatabaseDAO(config)
    if database_type == "sqlite":
        return DatabaseDAO(config["DATABASE_PATH"])
    raise ValueError("DATABASE_TYPE 仅支持 sqlite 或 mysql。")
