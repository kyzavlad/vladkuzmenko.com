import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modules
from app.api.routes import api_bp
from app.core.config import Config

# Create uploads directory if it doesn't exist
os.makedirs(os.getenv('UPLOAD_FOLDER', './uploads'), exist_ok=True)

def create_app(config_class=Config):
    app = Flask(__name__, 
                static_folder='./frontend/build/static',
                template_folder='./frontend/build')
    app.config.from_object(config_class)
    
    # Register CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Serve React App
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.template_folder, path)):
            return send_from_directory(app.template_folder, path)
        return send_from_directory(app.template_folder, 'index.html')
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"status": "error", "message": "Resource not found"}), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"status": "error", "message": "Internal server error"}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=os.getenv('DEBUG', 'False') == 'True') 