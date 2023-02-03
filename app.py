
from flask import Flask, request
import config as config
from functools import wraps
from ml import ProblemFixService
import pandas as pd

app = Flask(__name__)
app.config.from_object(config)

api_token_header = 'X-ACCESS-TOKEN'


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
    problems = request.get_json()['data']
    df = pd.DataFrame.from_dict(problems)
    df.columns = ['ID', 'Title', 'Problem']

    service = ProblemFixService()
    service.add_to_index(df)

    return {
        'message': 'Added records to index'
    }, 200
