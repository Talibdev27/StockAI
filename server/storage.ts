import { type User, type InsertUser, users } from "@shared/schema";
import { drizzle } from "drizzle-orm/neon-http";
import { neon } from "@neondatabase/serverless";
import { eq } from "drizzle-orm";

// modify the interface with any CRUD methods
// you might need

export interface IStorage {
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
}

// Initialize Neon database connection lazily
let db: ReturnType<typeof drizzle> | null = null;

const getDb = () => {
  if (!db) {
    const databaseUrl = process.env.DATABASE_URL;
    if (!databaseUrl) {
      throw new Error(
        "DATABASE_URL environment variable is not set. Please configure your Neon database connection string in .env file."
      );
    }
    const sql = neon(databaseUrl);
    db = drizzle(sql);
  }
  return db;
};

export class PostgresStorage implements IStorage {
  async getUser(id: string): Promise<User | undefined> {
    try {
      const database = getDb();
      const result = await database.select().from(users).where(eq(users.id, id)).limit(1);
      return result[0];
    } catch (error) {
      console.error("Error getting user by id:", error);
      throw error;
    }
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    try {
      const database = getDb();
      const result = await database
        .select()
        .from(users)
        .where(eq(users.username, username))
        .limit(1);
      return result[0];
    } catch (error) {
      console.error("Error getting user by username:", error);
      throw error;
    }
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    try {
      const database = getDb();
      const result = await database.insert(users).values(insertUser).returning();
      if (!result[0]) {
        throw new Error("Failed to create user");
      }
      return result[0];
    } catch (error) {
      console.error("Error creating user:", error);
      throw error;
    }
  }
}

// Fallback to MemStorage if DATABASE_URL is not set (for development/testing)
class MemStorage implements IStorage {
  private users: Map<string, User>;

  constructor() {
    this.users = new Map();
  }

  async getUser(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const { randomUUID } = await import("crypto");
    const id = randomUUID();
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }
}

// Use PostgreSQL storage if DATABASE_URL is set and not empty, otherwise fallback to MemStorage
export const storage = process.env.DATABASE_URL && process.env.DATABASE_URL.trim() !== ""
  ? new PostgresStorage()
  : new MemStorage();
