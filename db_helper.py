import sqlite3
from sqlite3 import Error
import threading

lock = threading.Lock()

class DbHelper:

	def connect_to_db():
		database = r"./data.db"
		
		sql_create_models_table = """CREATE TABLE IF NOT EXISTS cwd_frameworks (
                                    _id integer PRIMARY KEY,
                                    framework_name text NOT NULL UNIQUE,
									framework_object blob NOT NULL
                                );"""

		sql_create_rl_dataset_table = """CREATE TABLE IF NOT EXISTS rl_dataset (
                                    _id integer PRIMARY KEY,
									signal_key text NOT NULL,
                                    training_data blob NOT NULL
                                );"""
		
		sql_create_lstm_dataset_table = """CREATE TABLE IF NOT EXISTS lstm_dataset (
                                    _id integer PRIMARY KEY,
									framework_name text NOT NULL,
                                    training_data_seqs blob NOT NULL,

									FOREIGN KEY(framework_name) REFERENCES cwd_frameworks(framework_name) ON DELETE CASCADE
                                );"""
		
		""" create a database connection to a SQLite database """
		conn = DbHelper.create_connection(database)
		
		# create tables
		if conn is not None:
			# create tables
			DbHelper.create_table(sql_create_models_table, conn)
			DbHelper.create_table(sql_create_rl_dataset_table, conn)
			DbHelper.create_table(sql_create_lstm_dataset_table, conn)
		else:
			print("Error! cannot create the database connection.")

		return conn
		

	def update_record(self, table_name, columns, values, row_id, delegate_func):
		self.connect_to_db()
		self.update_by_id(table_name, columns, [values, ], row_id)
		
		# Return the id of the new records to the calling class using the delegate
		if delegate_func != None:
			delegate_func(None, values)
		return (row_id, values)


	def delete_record(self, table_name, row_id):
		self.connect_to_db()
		self.delete_by_id(table_name, row_id)

			
	def create_connection(db_file):
		""" create a database connection to the SQLite database
			specified by db_file
		:param db_file: database file
		:return: Connection object or None
		"""
		conn = None
		try:
			conn = sqlite3.connect(db_file)
			return conn
		except Error as e:
			#print(e)
			e = None
	 
		return conn
	
	
	def create_table(create_table_sql, conn):
		""" create a table from the create_table_sql statement
		:param conn: Connection object
		:param create_table_sql: a CREATE TABLE statement
		:return:
		"""
		try:
			cur = conn.cursor()
			cur.execute(create_table_sql)
		except Error as e:
			#print(e)
			e = None

	
	def insert(sql_command, command_args = []):
		# Connect to the database
		conn = DbHelper.connect_to_db()

		# Execute the sql command
		cur = conn.cursor()
		cur.execute(sql_command, command_args)
		
		# Save (commit) the changes
		conn.commit()

		# We can also close the connection if we are done with it.
		# Just be sure any changes have been committed or they will be lost.
		cur.close()
		conn.close()

		return cur.lastrowid
	
	
	def query(sql_command, command_args = [], keep_conn = False):
		# Connect to the database
		conn = DbHelper.connect_to_db()

		# Execute the sql command
		cur = conn.cursor()
		cur.execute(sql_command, command_args)
		
		rows = cur.fetchall()
		
		cur.close()
		if not(keep_conn):
			conn.close()
		
		return rows
	
	
	def update(sql_command, command_args = []):
		# Connect to the database
		conn = DbHelper.connect_to_db()

		# Execute the sql command
		cur = conn.cursor()
		cur.execute(sql_command, command_args)
		# Save (commit) the changes
		conn.commit()

		# Close connection
		cur.close()
		conn.close()

		# Return the updated row id
		return cur.lastrowid
	

	def delete_by_id(self, table_name, row_id):
		# Connect to the database
		conn = DbHelper.connect_to_db()

		# Create the command string
		sql = "DELETE from " + table_name \
			+ " WHERE _id = ? "
		
		# Execute the sql command
		cur = DbHelper._conn.cursor()
		cur.execute(sql, (row_id,))
		# Save (commit) the changes
		conn.commit()

		# Close connection
		cur.close()
		conn.close()