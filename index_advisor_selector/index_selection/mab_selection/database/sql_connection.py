import psycopg2
import pyodbc
import configparser

import constants


def get_sql_connection(args, db_file):
    """
    This method simply returns the sql connection based on the DB type
    and the connection settings defined in the `db.conf`.
    :return: connection
    """
    # Reading the Database configurations
    db_config = configparser.ConfigParser()
    # db_config.read(constants.ROOT_DIR + constants.DB_CONFIG)
    db_config.read(db_file)
    db_type = db_config["SYSTEM"]["db_type"]

    # (0731): newly added.
    if db_type == "MSSQL":
        server = db_config[db_type]["server"]
        database = db_config[db_type]["database"]
        driver = db_config[db_type]["driver"]

        return pyodbc.connect(
            r"Driver=" + driver + ";Server=" + server + ";Database=" + database + ";Trusted_Connection=yes;")

    elif db_type == "postgresql":
        host = db_config[db_type]["host"]

        # (1030): newly added.
        db_name = db_config[db_type]["database"]
        if args.db_name is not None:
            db_name = args.db_name

        port = db_config[db_type]["port"]
        user = db_config[db_type]["user"]
        password = db_config[db_type]["password"]

        connection = psycopg2.connect(host=host, database=db_name, port=port,
                                      user=user, password=password)
        connection.autocommit = True

        return connection

    else:
        raise NotImplementedError


def close_sql_connection(connection):
    """
    Take care of the closing process of the SQL connection
    :param connection: sql_connection
    :return: operation status
    """
    return connection.close()
