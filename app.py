
from flask import Flask, request
import config as config
import click
from functools import wraps
from ml import ProblemFixService, MYSQLService
import pandas as pd
import logging

app = Flask(__name__)
app.config.from_object(config)

api_token_header = 'X-ACCESS-TOKEN'

logging.basicConfig(filename='app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')

# CLI commands


@app.cli.command()
def index_from_mysql():
    try:
        print('Pushing to pinecone from db')
        problem_fix_service = ProblemFixService()
        mysql_service = MYSQLService()
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


# Authentication decorator
def requires_token(f):
    @wraps(f)
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


@app.route('/', methods=['GET'])
def index():
    return {
        'message': 'AutoInsider Problem Fix ML Service API',
    }


@app.route('/similar', methods=['GET'])
@requires_token
def similar():
    question = request.args.get('question')
    requested_limit = request.args.get('limit', type=int)

    if not requested_limit or requested_limit > 10:
        requested_limit = 5

    if not question:
        return {'message': 'Client error: No question provided'}, 400

    try:
        service = ProblemFixService()
        similarQuestions = service.get_similar_for(question, requested_limit)

        return {
            'message': 'AutoInsider Problem Fix ML Service',
            'question': question,
            'requested': requested_limit,
            'similar-questions': similarQuestions
        }, 200
    except:
        return {'message': 'Server error'}, 500


@app.route('/store', methods=['POST'])
@requires_token
def store_in_index():
    try:
        problems = request.get_json()['data']
        df = pd.DataFrame.from_dict(problems)
        df.columns = ['ID', 'Title', 'Context']

        service = ProblemFixService()
        records_indexed = service.add_to_index(df)

        return {
            'message': 'Added records to index',
            'records': records_indexed
        }, 200
    except Exception as err:
        logging.warning(f"Unexpected {err=}, {type(err)=}")
        return {'message': f"Unexpected {err=}, {type(err)=}"}, 500


if __name__ == "__main__":
    app.run(host='0.0.0.0')
