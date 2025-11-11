# Database Setup Guide

## Neon PostgreSQL Database Configuration

This project uses Neon PostgreSQL for persistent data storage. The Node.js backend will automatically use PostgreSQL when `DATABASE_URL` is configured.

### Setup Steps

1. **Create a Neon Database**
   - Go to [Neon Console](https://console.neon.tech/)
   - Create a new project and database
   - Copy the connection string (it looks like: `postgresql://username:password@hostname/database?sslmode=require`)

2. **Configure Environment Variable**
   - Create a `.env` file in the project root (if it doesn't exist)
   - Add your Neon database URL:
     ```
     DATABASE_URL=postgresql://username:password@hostname/database?sslmode=require
     ```

3. **Push Schema to Database**
   - Run the following command to create the necessary tables:
     ```bash
     npm run db:push
     ```
   - This will create the `users` table in your Neon database

4. **Start the Server**
   - The server will automatically use PostgreSQL storage when `DATABASE_URL` is set
   - If `DATABASE_URL` is not set, it will fallback to in-memory storage (data will be lost on restart)

### Python Backend

The Python backend currently uses SQLite (`predictions.db`) for storing predictions. This is separate from the Neon database and will continue to work independently.

If you want to migrate Python backend to PostgreSQL later, you can:
- Update `python_backend/evaluation.py` to use PostgreSQL
- Install `psycopg2` or `asyncpg` for Python PostgreSQL connection
- Update connection functions to use `DATABASE_URL`

### Troubleshooting

- **Error: DATABASE_URL environment variable is not set**
  - Make sure you've created a `.env` file with your Neon database URL
  
- **Error: Connection failed**
  - Verify your Neon database URL is correct
  - Check that your Neon database is running and accessible
  - Ensure SSL mode is set correctly (`?sslmode=require`)

- **Schema not found**
  - Run `npm run db:push` to create tables in your Neon database

