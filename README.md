# SDD1003-Backend

## FastAPI Backend

### Virtual Environment Setup
1. Create a virtual environment:
   ```bash
   python -m venv env
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source ./env/bin/activate
     ```

3. Install dependencies inside the virtual environment:
   ```bash
  pip install fastapi uvicorn pymongo python-dotenv
   ```

### Folder Structure
- `main.py`: Entry point of the application.
- `app/`
  - `__init__.py`: Marks the folder as a Python package.
  - `database.py`: MongoDB connection logic.
  - `routes.py`: API routes.

### Run the Backend

To start the FastAPI backend server, run the following command in your console (from the backend folder):

```bash
uvicorn main:app --reload
```

- The server will be available at: http://127.0.0.1:8000
- The interactive API docs are at: http://127.0.0.1:8000/docs
- The alternative docs are at: http://127.0.0.1:8000/redoc

Make sure your virtual environment is activated before running the command.