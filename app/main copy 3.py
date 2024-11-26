from fastapi import FastAPI, Request, Form, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
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

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    # Verificar credenciales usando db_manager
    user = db_manager.verify_credentials(username, password)
    
    if user:
        return templates.TemplateResponse(
            "index.html",
            {"request": request}
        )
    else:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Credenciales inválidas"
            }
        )
    
embeddings_processor = EmbeddingsProcessor()

@app.post("/generate_embeddings")
async def generate_embeddings(request: Request):
    try:
        # Recibir los datos JSON del cuerpo de la solicitud
        data = await request.json()
        print("Datos recibidos:", data)
        
        if not isinstance(data, dict):
            raise ValueError(f"Se esperaba un diccionario, se recibió {type(data)}")
        
        if 'cited_document_id' not in data:
            raise ValueError("El JSON debe contener la clave 'cited_document_id'")
        
        # Procesar los embeddings
        result = embeddings_processor.process_patent_data(data)
        print("Procesamiento exitoso")
        
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
'''
@app.post("/generate_embeddings")
async def generate_embeddings(request: Request):
    try:
        # Recibir los datos JSON del cuerpo de la solicitud
        data = await request.json()
        print('---------- Data para embeddings ----------')
        print(data)
        
        # Procesar los embeddings
        result = embeddings_processor.process_patent_data(data)
        print('---------- Embeddings obtenidos ----------')
        print(result)
        print('-----------------------------------------')
        
        return JSONResponse(content=result)
    except Exception as e:
        print("Error en el servidor:", str(e))  # Debug
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )    
'''        