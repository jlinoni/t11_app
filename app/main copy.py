from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from datetime import datetime
from .database.db_manager import DatabaseManager

app = FastAPI()

# Configurar archivos estáticos y templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Inicializar el gestor de base de datos
db_manager = DatabaseManager()

# Variables globales
MAX_LOGIN_ATTEMPTS = 3

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": None}
    )

@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    # Verificar intentos de login
    attempts_info = db_manager.check_login_attempts(username)
    
    if attempts_info and attempts_info[0] >= MAX_LOGIN_ATTEMPTS:
        last_attempt = datetime.strptime(attempts_info[1], '%Y-%m-%d %H:%M:%S')
        if (datetime.now() - last_attempt).seconds < 3600:  # 1 hora de bloqueo
            return templates.TemplateResponse(
                "login.html",
                {
                    "request": request,
                    "error": "Cuenta bloqueada. Intente más tarde."
                }
            )
        else:
            # Resetear intentos después de 1 hora
            db_manager.reset_login_attempts(username)
    
    # Verificar credenciales
    user = db_manager.verify_credentials(username, password)
    
    if user:
        # Resetear intentos de login exitosos
        db_manager.reset_login_attempts(username)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "settings": {
                    "title": "Sistema de Análisis de Patentes",
                    "subtitle": "Visualización y Análisis",
                    "short_description": "Herramienta para análisis de patentes",
                    "icon_path": "/static/img/icon.png",
                    "graph_types": ["Visualización de Puntos", "Gráfico de Barras", "Mapa de Calor"],
                    "sub_options": {
                        "Visualización de Puntos": ["2D", "3D", "Temporal"]
                    },
                    "research_text": "Texto de investigación",
                    "researchers": ["Investigador 1", "Investigador 2"],
                    "institute": "Instituto de Investigación"
                }
            }
        )
    else:
        # Incrementar intentos fallidos
        current_attempts = db_manager.increment_login_attempts(username)
        
        if current_attempts >= MAX_LOGIN_ATTEMPTS:
            error_message = "Ha excedido el número máximo de intentos. Cuenta bloqueada."
        else:
            remaining = MAX_LOGIN_ATTEMPTS - current_attempts
            error_message = f"Credenciales inválidas. Intentos restantes: {remaining}"
        
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": error_message}
        )