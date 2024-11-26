from fastapi import FastAPI, Request, Form, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from .database.db_manager import DatabaseManager
from .embeddings import EmbeddingsProcessor
import json

app = FastAPI()

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configurar templates
templates = Jinja2Templates(directory="app/templates")

# Inicializar gestor de base de datos
db_manager = DatabaseManager()

# Diccionario para mantener las instancias de EmbeddingsProcessor por sesión
embeddings_processors = {}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = db_manager.verify_credentials(username, password)
    
    if user:
        # Crear una nueva instancia de EmbeddingsProcessor para esta sesión
        session_id = str(id(request))
        embeddings_processors[session_id] = EmbeddingsProcessor()
        
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "session_id": session_id}
        )
    else:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Credenciales inválidas"
            }
        )

@app.post("/generate_embeddings")
async def generate_embeddings(request: Request):
    try:
        # Obtener el ID de sesión
        session_id = str(id(request))
        
        # Obtener o crear el procesador de embeddings para esta sesión
        if session_id not in embeddings_processors:
            embeddings_processors[session_id] = EmbeddingsProcessor()
        
        # Recibir los datos JSON del cuerpo de la solicitud
        data = await request.json()
        print(f"Datos recibidos para sesión {session_id}:", data)
        
        if not isinstance(data, dict):
            raise ValueError(f"Se esperaba un diccionario, se recibió {type(data)}")
        
        if 'cited_document_id' not in data:
            raise ValueError("El JSON debe contener la clave 'cited_document_id'")
        
        # Procesar los embeddings usando el procesador de esta sesión
        result = embeddings_processors[session_id].process_patent_data(data)
        print(f"Procesamiento exitoso para sesión {session_id}")
        
        return JSONResponse(content=result)
    except json.JSONDecodeError as e:
        print(f"Error decodificando JSON: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": "JSON inválido", "details": str(e)}
        )
    except Exception as e:
        print(f"Error procesando embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Error procesando embeddings", "details": str(e)}
        )

# Opcional: Limpiar procesadores antiguos periódicamente
@app.on_event("startup")
async def startup_event():
    # Crear el directorio base de caché si no existe
    Path("data/embeddings_cache").mkdir(parents=True, exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    # Limpiar los procesadores al cerrar la aplicación
    embeddings_processors.clear()

@app.post("/clear_session")
async def clear_session(request: Request):
    try:
        session_id = str(id(request))
        if session_id in embeddings_processors:
            del embeddings_processors[session_id]
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/reset_view", response_class=HTMLResponse)
async def reset_view(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "reset": True}
    )    