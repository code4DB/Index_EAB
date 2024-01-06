import os
import json
import logging
import importlib
import configparser

from .workload import Column, Table
from .postgres_dbms import PostgresDatabaseConnector


class Schema(object):
    def __init__(self, db_config_file, schema_file=None, filters={},
                 user=None, password=None, host=None, db_name=None, port=None):
        self.db_config = configparser.ConfigParser()
        self.db_config.read(db_config_file)

        if user is not None:
            self.db_config["postgresql"]["user"] = user
        if password is not None:
            self.db_config["postgresql"]["password"] = password
        if host is not None:
            self.db_config["postgresql"]["host"] = host
        if db_name is not None:
            self.db_config["postgresql"]["database"] = db_name
        if port is not None:
            self.db_config["postgresql"]["port"] = port

        # get the database info
        self.schema_file = schema_file
        if self.schema_file is None or not os.path.exists(self.schema_file):
            db_connector = PostgresDatabaseConnector(self.db_config, autocommit=True)
            tables, columns = self.get_columns_from_db(db_connector)
        else:
            tables, columns = self.get_columns_from_schema(self.schema_file)

        self.database_name = self.db_config["postgresql"]["database"]
        self.tables = tables
        self.columns = columns

        # self.columns = []
        # for table in self.tables:
        #     for column in table.columns:
        #         self.columns.append(column)

        # `column_filters`
        for filter_name in filters.keys():
            filter_class = getattr(importlib.import_module("swirl_selection.swirl_utils.schema"), filter_name)
            filter_instance = filter_class(filters[filter_name], self.db_config)
            self.columns = filter_instance.apply_filter(self.columns)

    def get_columns_from_db(self, db_connector):
        db_connector.create_connection()

        tables, columns = list(), list()
        for table in db_connector.get_tables():
            table_object = Table(table)
            tables.append(table_object)
            for col in db_connector.get_cols(table):
                column_object = Column(col)
                table_object.add_column(column_object)
                columns.append(column_object)
        db_connector.close()

        return tables, columns

    def get_columns_from_schema(self, schema_file):
        tables, columns = list(), list()
        with open(schema_file, "r") as rf:
            db_schema = json.load(rf)

        for item in db_schema:
            table_object = Table(item["table"])
            tables.append(table_object)
            for col_info in item["columns"]:
                column_object = Column(col_info["name"])
                table_object.add_column(column_object)
                columns.append(column_object)

        return tables, columns


class TableNumRowsFilter(object):
    def __init__(self, threshold, db_config):
        self.threshold = threshold
        self.connector = PostgresDatabaseConnector(db_config, autocommit=True)
        self.connector.create_statistics()

    def apply_filter(self, columns):
        """
        Filter the columns according to the number of rows in the table.
        :param columns:
        :return:
        """
        output_columns = []

        for column in columns:
            table_name = column.table.name
            table_num_rows = self.connector.exec_fetch(
                f"SELECT reltuples::bigint AS estimate FROM pg_class where relname='{table_name}'"
            )[0]

            if table_num_rows > self.threshold:
                output_columns.append(column)
        # : newly added.
        self.connector.close()

        logging.debug("call the `apply_filter` function in `TableNumRowsFilter` class.")
        logging.warning(f"Reduced columns from {len(columns)} to {len(output_columns)}.")

        return output_columns
