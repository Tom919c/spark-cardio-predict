from src.controllers.analysis_controller import analysis_bp
from src.controllers.data_controller import data_bp
from src.controllers.health_controller import health_bp
from src.controllers.page_controller import pages_bp
from src.controllers.task_controller import task_bp
from src.controllers.upload_controller import upload_bp
from src.controllers.dataset_controller import dataset_bp
from src.controllers.risk_controller import risk_bp
from src.controllers.risk_phase2_controller import risk_phase2_bp
from src.controllers.trend_controller import trend_bp


def register_blueprints(app):
    app.register_blueprint(health_bp)
    app.register_blueprint(pages_bp)
    app.register_blueprint(data_bp, url_prefix="/api/data")
    app.register_blueprint(risk_bp, url_prefix="/api/risk")
    app.register_blueprint(risk_phase2_bp, url_prefix="/api/risk")
    app.register_blueprint(upload_bp, url_prefix="/api/upload")
    app.register_blueprint(upload_bp, url_prefix="/api/uploads", name_prefix="plural")
    app.register_blueprint(task_bp, url_prefix="/api/tasks")
    app.register_blueprint(task_bp, url_prefix="/api/task", name_prefix="singular")
    app.register_blueprint(dataset_bp, url_prefix="/api/datasets")
    app.register_blueprint(trend_bp, url_prefix="/api/trends")
    app.register_blueprint(analysis_bp, url_prefix="/api/analysis")
