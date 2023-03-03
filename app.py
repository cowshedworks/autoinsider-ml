
from flask import Flask, request
import config
from functools import wraps
from ml import EuropeanRailGuideService, AutoInsiderService, MYSQLService
import pandas as pd
import logging

app = Flask(__name__)
app.config.from_object(config)

api_token_header = 'X-ACCESS-TOKEN'

logging.basicConfig(filename='app.log',
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

# CLI commands


@ app.cli.command('index_ai_from_mysql')
def index_ai_from_mysql():
    try:
        print('Pushing AutoInsider problems to pinecone')
        problem_fix_service = AutoInsiderService()
        mysql_service = MYSQLService(
            host=app.config["MYSQL_HOST"],
            user=app.config["MYSQL_USER"],
            password=app.config["MYSQL_PASSWORD"],
            port=app.config["MYSQL_PORT"],
            database='autoinsider'
        )
        problem_fix_service.rebuild_index()
        df = mysql_service.get_all_problems()
        print(f"Attempting to index {len(df)} records")
        total_indexed = 0
        for i in range(0, len(df), 256):
            i_end = min(i+256, len(df))
            # extract batch
            batch = df.iloc[i:i_end]
            total_indexed += problem_fix_service.add_to_index(batch)
            print(f"Inserted {len(batch)} records")
        print(f"Completed, indexed {total_indexed} records")
    except Exception as err:
        print({'message': f"Unexpected {err=}, {type(err)=}"})


@ app.cli.command('index_erg_from_mysql')
def index_erg_from_mysql():
    try:
        print('Pushing European Rail Guide content to pinecone')
        erg_service = EuropeanRailGuideService()
        mysql_service = MYSQLService(
            host=app.config["MYSQL_HOST_HOMESTEAD"],
            user=app.config["MYSQL_USER"],
            password=app.config["MYSQL_PASSWORD"],
            port=app.config["MYSQL_PORT_HOMESTEAD"],
            database='erg'
        )
        # erg_service.rebuild_index()
        df = mysql_service.get_all_places()
        print(f"Attempting to index {len(df)} records")
        total_indexed = 0
        for i in range(0, len(df), 256):
            i_end = min(i+256, len(df))
            # extract batch
            batch = df.iloc[i:i_end]
            total_indexed += erg_service.add_to_index(batch)
            print(f"Inserted {len(batch)} records")
        print(f"Completed, indexed {total_indexed} records")
    except Exception as err:
        print({'message': f"Unexpected {err=}, {type(err)=}"})

# Authentication decorator


def requires_token(f):
    @ wraps(f)
    def decorator(*args, **kwargs):
        token = None
        if api_token_header in request.headers:
            token = request.headers[api_token_header]
        if not token:
            return {"message": "Client error: A valid token is missing!"}, 400
        if not app.config['API_TOKEN'] == token:
            return {"message": "Client error: Your token is invalid!"}, 401
        return f(*args, **kwargs)
    return decorator


@ app.route('/', methods=['GET'])
def index():
    return {
        'message': 'ML Service API',
    }


@app.route('/autoinsider/similar/problems/index', methods=['GET'])
@requires_token
def similar():
    question = request.args.get('question')
    requested_limit = request.args.get('limit', type=int)

    if not requested_limit or requested_limit > 10:
        requested_limit = 5

    if not question:
        return {'message': 'Client error: No question provided'}, 400

    service = AutoInsiderService()
    similarQuestions = service.get_similar_for(question, requested_limit)

    return {
        'message': 'AutoInsider Problem Fix ML Service',
        'question': question,
        'requested': requested_limit,
        'similar-questions': similarQuestions
    }, 200


@app.route('/autoinsider/similar/problems/store', methods=['POST'])
@requires_token
def store_in_index():
    try:
        problems = request.get_json()['data']
        df = pd.DataFrame.from_dict(problems)
        df.columns = ['ID', 'Title', 'Context']

        service = AutoInsiderService()
        records_indexed = service.add_to_index(df)

        return {
            'message': 'Added records to index',
            'records': records_indexed
        }, 200
    except Exception as err:
        return {'message': f"Unexpected {err=}, {type(err)=}"}, 500


@app.route('/autoinsider/similar/problems/delete', methods=['POST'])
@requires_token
def delete_from_index():
    try:
        vector_ids = request.get_json()['data']

        if type(vector_ids) is not list:
            raise TypeError("Should be a list of vector ids")

        service = AutoInsiderService()
        service.delete_from_index(vector_ids)

        return {
            'message': 'Deleted records from index',
            'records': len(vector_ids)
        }, 200
    except Exception as err:
        return {'message': f"Unexpected {err=}, {type(err)=}"}, 500


@app.route('/europeanrailguide/similar/places/index', methods=['GET'])
@requires_token
def erg_similar_places():
    query = request.args.get('query')
    requested_limit = request.args.get('limit', type=int)

    if not requested_limit or requested_limit > 10:
        requested_limit = 5

    if not query:
        return {'message': 'Client error: No query provided'}, 400

    service = EuropeanRailGuideService()
    similarPlaces = service.get_similar_for(query, requested_limit)

    return {
        'message': 'European Rail Guide Similar Place Service',
        'query': query,
        'requested': requested_limit,
        'similar-places': similarPlaces
    }, 200


@app.route('/europeanrailguide/similar/places/store', methods=['POST'])
@requires_token
def store_in_index():
    try:
        places = request.get_json()['data']
        df = pd.DataFrame.from_dict(places)
        df.columns = ['ID', 'Title', 'Context']

        service = EuropeanRailGuideService()
        records_indexed = service.add_to_index(df)

        return {
            'message': 'Added records to index',
            'records': records_indexed
        }, 200
    except Exception as err:
        return {'message': f"Unexpected {err=}, {type(err)=}"}, 500


@app.route('/europeanrailguide/similar/places/delete', methods=['POST'])
@requires_token
def delete_from_index():
    try:
        vector_ids = request.get_json()['data']

        if type(vector_ids) is not list:
            raise TypeError("Should be a list of vector ids")

        service = EuropeanRailGuideService()
        service.delete_from_index(vector_ids)

        return {
            'message': 'Deleted records from index',
            'records': len(vector_ids)
        }, 200
    except Exception as err:
        return {'message': f"Unexpected {err=}, {type(err)=}"}, 500


if __name__ == "__main__":
    app.run(host='0.0.0.0')
