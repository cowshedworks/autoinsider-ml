from flask import Flask, request
import config as config
from functools import wraps

app = Flask(__name__)
app.config.from_object(config)

api_token_header = 'X-ACCESS-TOKEN'


# Authentication decorator
def requires_token(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = None
        # ensure the jwt-token is passed with the headers
        if api_token_header in request.headers:
            token = request.headers[api_token_header]
        if not token:  # throw error if no token provided
            return {"message": "A valid token is missing!"}
        if not app.config['API_TOKEN'] == token:
            return {"message": "Your token is invalid!"}
        return f(*args, **kwargs)
    return decorator


@app.route('/', methods=['POST'])
@requires_token
def similar():
    return {
        'message': 'Welcome to the ML Service!'
    }
