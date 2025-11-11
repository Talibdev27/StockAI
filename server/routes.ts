import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";

// In production, if PYTHON_API_BASE is not set, assume Python backend runs on same host
const PYTHON_API_BASE = process.env.PYTHON_API_BASE || 
  (process.env.NODE_ENV === "production" ? "http://localhost:5001" : "http://localhost:5001");

export async function registerRoutes(app: Express): Promise<Server> {
  // Proxy routes to Python backend
  app.use("/api/predict", async (req, res) => {
    try {
      const response = await fetch(`${PYTHON_API_BASE}${req.url}`, {
        method: req.method,
        headers: {
          "Content-Type": "application/json",
        },
        body: req.method !== "GET" ? JSON.stringify(req.body) : undefined,
      });
      const data = await response.json();
      res.status(response.status).json(data);
    } catch (error) {
      res.status(500).json({ error: "Failed to connect to Python backend" });
    }
  });

  app.use("/api/backtest", async (req, res) => {
    try {
      const response = await fetch(`${PYTHON_API_BASE}${req.url}`, {
        method: req.method,
        headers: {
          "Content-Type": "application/json",
        },
        body: req.method !== "GET" ? JSON.stringify(req.body) : undefined,
      });
      const data = await response.json();
      res.status(response.status).json(data);
    } catch (error) {
      res.status(500).json({ error: "Failed to connect to Python backend" });
    }
  });

  app.use("/api/indicators", async (req, res) => {
    try {
      const response = await fetch(`${PYTHON_API_BASE}${req.url}`, {
        method: req.method,
        headers: {
          "Content-Type": "application/json",
        },
      });
      const data = await response.json();
      res.status(response.status).json(data);
    } catch (error) {
      res.status(500).json({ error: "Failed to connect to Python backend" });
    }
  });

  app.use("/api/quote", async (req, res) => {
    try {
      const response = await fetch(`${PYTHON_API_BASE}${req.url}`, {
        method: req.method,
        headers: {
          "Content-Type": "application/json",
        },
      });
      const data = await response.json();
      res.status(response.status).json(data);
    } catch (error) {
      res.status(500).json({ error: "Failed to connect to Python backend" });
    }
  });

  app.use("/api/historical", async (req, res) => {
    try {
      const response = await fetch(`${PYTHON_API_BASE}${req.url}`, {
        method: req.method,
        headers: {
          "Content-Type": "application/json",
        },
      });
      const data = await response.json();
      res.status(response.status).json(data);
    } catch (error) {
      res.status(500).json({ error: "Failed to connect to Python backend" });
    }
  });

  app.use("/api/predictions", async (req, res) => {
    try {
      const response = await fetch(`${PYTHON_API_BASE}${req.url}`, {
        method: req.method,
        headers: {
          "Content-Type": "application/json",
        },
        body: req.method !== "GET" ? JSON.stringify(req.body) : undefined,
      });
      const data = await response.json();
      res.status(response.status).json(data);
    } catch (error) {
      res.status(500).json({ error: "Failed to connect to Python backend" });
    }
  });

  app.use("/api/stocks", async (req, res) => {
    try {
      const response = await fetch(`${PYTHON_API_BASE}${req.url}`, {
        method: req.method,
        headers: {
          "Content-Type": "application/json",
        },
      });
      const data = await response.json();
      res.status(response.status).json(data);
    } catch (error) {
      res.status(500).json({ error: "Failed to connect to Python backend" });
    }
  });

  // put application routes here
  // prefix all routes with /api

  // use storage to perform CRUD operations on the storage interface
  // e.g. storage.insertUser(user) or storage.getUserByUsername(username)

  const httpServer = createServer(app);

  return httpServer;
}
