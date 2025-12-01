from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.estates_CRUD_routes import router_crud
from app.routes.estate_list_routes import router_list as router_list
from app.routes.batch_routes import router_batch as router_batch
from app.routes.ml_routes import router_ml as router_ml

# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes for estate listing and autocomplete
app.include_router(router_list)
# Include routes for estate CRUD operations
app.include_router(router_crud)
# Include routes for batch operations
app.include_router(router_batch)
# Include routes for ml operations
app.include_router(router_ml)
