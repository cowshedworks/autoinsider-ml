
from flask import Flask, request
import config as config
from functools import wraps
from ml import ProblemFixService

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


@app.route('/', methods=['POST'])
@requires_token
def similar():
    requestedLimit = 5
    question = request.form.get('question')

    if not question:
        return {'message': 'Client error: No question provided'}, 400

    try:
        service = ProblemFixService(app)
        similarQuestions = service.getSimilarFor(question, requestedLimit)

        return {
            'message': 'AutoInsider Problem Fix ML Service',
            'question': question,
            'requested': requestedLimit,
            'similar-questions': similarQuestions
        }, 200
    except:
        return {'message': 'Server error'}, 500
