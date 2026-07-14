"""根据配置创建 SQLite 或 MySQL 元数据库 DAO。"""

from src.dao.database_dao import DatabaseDAO


def create_database_dao(config):
    database_type = str(config.get("DATABASE_TYPE", "sqlite")).lower()
    if database_type == "mysql":
        from src.dao.mysql_dao import MySQLDatabaseDAO

        return MySQLDatabaseDAO(config)
    return DatabaseDAO(config["DATABASE_PATH"])
