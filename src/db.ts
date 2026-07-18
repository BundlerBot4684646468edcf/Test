import { DatabaseSync } from 'node:sqlite';
import crypto from 'crypto';
import path from 'path';

// SQLite database — a single local file, no server, no Docker, no downloads.
const DB_FILE = process.env.DB_FILE || path.join(process.cwd(), 'mundpost.db');
const db = new DatabaseSync(DB_FILE);

db.exec('PRAGMA journal_mode = WAL;');
db.exec('PRAGMA foreign_keys = ON;');

// --- Schema ---------------------------------------------------------------
db.exec(`
CREATE TABLE IF NOT EXISTS Business (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  googlePlaceId TEXT UNIQUE,
  googleReviewLink TEXT,
  ownerName TEXT NOT NULL,
  ownerPhotoUrl TEXT,
  timezone TEXT NOT NULL DEFAULT 'Europe/Rome',
  dailyBatchLimit INTEGER NOT NULL DEFAULT 20,
  createdAt TEXT NOT NULL,
  updatedAt TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Customer (
  id TEXT PRIMARY KEY,
  businessId TEXT NOT NULL REFERENCES Business(id) ON DELETE CASCADE,
  firstName TEXT NOT NULL,
  phone TEXT,
  email TEXT,
  servedAt TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'past',
  optOut INTEGER NOT NULL DEFAULT 0,
  createdAt TEXT NOT NULL,
  updatedAt TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_customer_business ON Customer(businessId);

CREATE TABLE IF NOT EXISTS ReviewRequest (
  id TEXT PRIMARY KEY,
  customerId TEXT NOT NULL REFERENCES Customer(id) ON DELETE CASCADE,
  businessId TEXT NOT NULL REFERENCES Business(id) ON DELETE CASCADE,
  channel TEXT NOT NULL DEFAULT 'sms',
  status TEXT NOT NULL DEFAULT 'queued',
  scheduledAt TEXT,
  sentAt TEXT,
  remindedAt TEXT,
  reviewedAt TEXT,
  createdAt TEXT NOT NULL,
  updatedAt TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rr_business ON ReviewRequest(businessId);
CREATE INDEX IF NOT EXISTS idx_rr_customer ON ReviewRequest(customerId);
CREATE INDEX IF NOT EXISTS idx_rr_status ON ReviewRequest(status);

CREATE TABLE IF NOT EXISTS ReviewEvent (
  id TEXT PRIMARY KEY,
  businessId TEXT NOT NULL REFERENCES Business(id) ON DELETE CASCADE,
  newReviewCount INTEGER NOT NULL DEFAULT 0,
  avgRating REAL,
  totalReviews INTEGER NOT NULL DEFAULT 0,
  checkedAt TEXT NOT NULL,
  createdAt TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_re_business ON ReviewEvent(businessId);
`);

// --- Lightweight migrations (add columns to existing tables) ---------------
function ensureColumn(table: string, column: string, ddl: string) {
  const cols = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{
    name: string;
  }>;
  if (!cols.some((c) => c.name === column)) {
    db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${ddl};`);
  }
}
// Sign area on the owner photo (relative 0..1 coords) for name personalization
ensureColumn('Business', 'signX', 'REAL');
ensureColumn('Business', 'signY', 'REAL');
ensureColumn('Business', 'signWidth', 'REAL');
ensureColumn('Business', 'signHeight', 'REAL');
ensureColumn('Business', 'signRotation', 'REAL');

// --- Helpers --------------------------------------------------------------
export function newId(): string {
  return crypto.randomUUID();
}
export function nowISO(): string {
  return new Date().toISOString();
}

// Types
export interface Business {
  id: string;
  name: string;
  googlePlaceId: string | null;
  googleReviewLink: string | null;
  ownerName: string;
  ownerPhotoUrl: string | null;
  timezone: string;
  dailyBatchLimit: number;
  signX: number | null;
  signY: number | null;
  signWidth: number | null;
  signHeight: number | null;
  signRotation: number | null;
  createdAt: string;
  updatedAt: string;
}
export interface Customer {
  id: string;
  businessId: string;
  firstName: string;
  phone: string | null;
  email: string | null;
  servedAt: string;
  source: string;
  optOut: number; // 0 | 1
  createdAt: string;
  updatedAt: string;
}
export interface ReviewRequest {
  id: string;
  customerId: string;
  businessId: string;
  channel: string;
  status: string;
  scheduledAt: string | null;
  sentAt: string | null;
  remindedAt: string | null;
  reviewedAt: string | null;
  createdAt: string;
  updatedAt: string;
}
export interface ReviewEvent {
  id: string;
  businessId: string;
  newReviewCount: number;
  avgRating: number | null;
  totalReviews: number;
  checkedAt: string;
  createdAt: string;
}

// --- Business repo --------------------------------------------------------
export const businesses = {
  create(data: { name: string; ownerName: string; timezone?: string }): Business {
    const id = newId();
    const ts = nowISO();
    db.prepare(
      `INSERT INTO Business (id, name, ownerName, timezone, createdAt, updatedAt)
       VALUES (?, ?, ?, ?, ?, ?)`
    ).run(id, data.name, data.ownerName, data.timezone || 'Europe/Rome', ts, ts);
    return this.get(id)!;
  },
  get(id: string): Business | undefined {
    return db.prepare('SELECT * FROM Business WHERE id = ?').get(id) as
      | Business
      | undefined;
  },
  update(id: string, fields: Partial<Business>): Business | undefined {
    const allowed = [
      'name',
      'ownerName',
      'timezone',
      'googlePlaceId',
      'googleReviewLink',
      'ownerPhotoUrl',
      'dailyBatchLimit',
      'signX',
      'signY',
      'signWidth',
      'signHeight',
      'signRotation',
    ];
    const keys = Object.keys(fields).filter((k) => allowed.includes(k));
    if (keys.length > 0) {
      const setClause = keys.map((k) => `${k} = ?`).join(', ');
      const values = keys.map((k) => (fields as any)[k]);
      db.prepare(
        `UPDATE Business SET ${setClause}, updatedAt = ? WHERE id = ?`
      ).run(...values, nowISO(), id);
    }
    return this.get(id);
  },
  listWithPlaceId(): Business[] {
    return db
      .prepare('SELECT * FROM Business WHERE googlePlaceId IS NOT NULL')
      .all() as unknown as Business[];
  },
};

// --- Customer repo --------------------------------------------------------
export const customers = {
  create(data: {
    businessId: string;
    firstName: string;
    phone?: string | null;
    email?: string | null;
    servedAt: string;
    source?: string;
  }): Customer {
    const id = newId();
    const ts = nowISO();
    db.prepare(
      `INSERT INTO Customer (id, businessId, firstName, phone, email, servedAt, source, createdAt, updatedAt)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(
      id,
      data.businessId,
      data.firstName,
      data.phone ?? null,
      data.email ?? null,
      data.servedAt,
      data.source || 'past',
      ts,
      ts
    );
    return this.get(id)!;
  },
  get(id: string): Customer | undefined {
    return db.prepare('SELECT * FROM Customer WHERE id = ?').get(id) as
      | Customer
      | undefined;
  },
  list(businessId: string, skip: number, take: number): Customer[] {
    return db
      .prepare(
        'SELECT * FROM Customer WHERE businessId = ? ORDER BY createdAt DESC LIMIT ? OFFSET ?'
      )
      .all(businessId, take, skip) as unknown as Customer[];
  },
  count(businessId: string): number {
    const row = db
      .prepare('SELECT COUNT(*) AS c FROM Customer WHERE businessId = ?')
      .get(businessId) as { c: number };
    return row.c;
  },
  update(id: string, fields: Partial<Customer>): Customer | undefined {
    const allowed = ['firstName', 'phone', 'email', 'servedAt', 'source', 'optOut'];
    const keys = Object.keys(fields).filter((k) => allowed.includes(k));
    if (keys.length > 0) {
      const setClause = keys.map((k) => `${k} = ?`).join(', ');
      const values = keys.map((k) => {
        const v = (fields as any)[k];
        return typeof v === 'boolean' ? (v ? 1 : 0) : v;
      });
      db.prepare(
        `UPDATE Customer SET ${setClause}, updatedAt = ? WHERE id = ?`
      ).run(...values, nowISO(), id);
    }
    return this.get(id);
  },
  delete(id: string): void {
    db.prepare('DELETE FROM Customer WHERE id = ?').run(id);
  },
};

// --- ReviewRequest repo ---------------------------------------------------
export const reviewRequests = {
  create(data: {
    customerId: string;
    businessId: string;
    channel?: string;
    status?: string;
    scheduledAt?: string | null;
  }): ReviewRequest {
    const id = newId();
    const ts = nowISO();
    db.prepare(
      `INSERT INTO ReviewRequest (id, customerId, businessId, channel, status, scheduledAt, createdAt, updatedAt)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(
      id,
      data.customerId,
      data.businessId,
      data.channel || 'sms',
      data.status || 'queued',
      data.scheduledAt ?? null,
      ts,
      ts
    );
    return this.get(id)!;
  },
  get(id: string): ReviewRequest | undefined {
    return db.prepare('SELECT * FROM ReviewRequest WHERE id = ?').get(id) as
      | ReviewRequest
      | undefined;
  },
  findActiveForCustomer(customerId: string): ReviewRequest | undefined {
    return db
      .prepare(
        `SELECT * FROM ReviewRequest
         WHERE customerId = ? AND status IN ('queued','sent','reminded')
         LIMIT 1`
      )
      .get(customerId) as ReviewRequest | undefined;
  },
  list(
    businessId: string,
    opts: { status?: string; skip: number; take: number }
  ): ReviewRequest[] {
    if (opts.status) {
      return db
        .prepare(
          `SELECT * FROM ReviewRequest WHERE businessId = ? AND status = ?
           ORDER BY createdAt DESC LIMIT ? OFFSET ?`
        )
        .all(businessId, opts.status, opts.take, opts.skip) as unknown as ReviewRequest[];
    }
    return db
      .prepare(
        `SELECT * FROM ReviewRequest WHERE businessId = ?
         ORDER BY createdAt DESC LIMIT ? OFFSET ?`
      )
      .all(businessId, opts.take, opts.skip) as unknown as ReviewRequest[];
  },
  count(businessId: string, status?: string): number {
    if (status) {
      const row = db
        .prepare(
          'SELECT COUNT(*) AS c FROM ReviewRequest WHERE businessId = ? AND status = ?'
        )
        .get(businessId, status) as { c: number };
      return row.c;
    }
    const row = db
      .prepare('SELECT COUNT(*) AS c FROM ReviewRequest WHERE businessId = ?')
      .get(businessId) as { c: number };
    return row.c;
  },
  update(id: string, fields: Partial<ReviewRequest>): ReviewRequest | undefined {
    const allowed = [
      'channel',
      'status',
      'scheduledAt',
      'sentAt',
      'remindedAt',
      'reviewedAt',
    ];
    const keys = Object.keys(fields).filter((k) => allowed.includes(k));
    if (keys.length > 0) {
      const setClause = keys.map((k) => `${k} = ?`).join(', ');
      const values = keys.map((k) => (fields as any)[k]);
      db.prepare(
        `UPDATE ReviewRequest SET ${setClause}, updatedAt = ? WHERE id = ?`
      ).run(...values, nowISO(), id);
    }
    return this.get(id);
  },
  optOutForCustomer(customerId: string): void {
    db.prepare(
      `UPDATE ReviewRequest SET status = 'opted_out', updatedAt = ?
       WHERE customerId = ? AND status IN ('queued','sent')`
    ).run(nowISO(), customerId);
  },
  // Queued requests whose scheduled time (if any) has arrived.
  listDue(limit: number): ReviewRequest[] {
    return db
      .prepare(
        `SELECT * FROM ReviewRequest
         WHERE status = 'queued'
           AND (scheduledAt IS NULL OR scheduledAt <= ?)
         ORDER BY createdAt ASC LIMIT ?`
      )
      .all(nowISO(), limit) as unknown as ReviewRequest[];
  },
  countSentToday(businessId: string, startOfDayISO: string): number {
    const row = db
      .prepare(
        `SELECT COUNT(*) AS c FROM ReviewRequest
         WHERE businessId = ? AND status IN ('sent','reminded','reviewed')
           AND sentAt >= ?`
      )
      .get(businessId, startOfDayISO) as { c: number };
    return row.c;
  },
  listReminderCandidates(cutoffISO: string, limit: number): ReviewRequest[] {
    return db
      .prepare(
        `SELECT * FROM ReviewRequest
         WHERE status = 'sent' AND remindedAt IS NULL AND sentAt <= ?
         LIMIT ?`
      )
      .all(cutoffISO, limit) as unknown as ReviewRequest[];
  },
};

// --- ReviewEvent repo -----------------------------------------------------
export const reviewEvents = {
  create(data: {
    businessId: string;
    newReviewCount: number;
    avgRating: number | null;
    totalReviews: number;
    checkedAt: string;
  }): ReviewEvent {
    const id = newId();
    db.prepare(
      `INSERT INTO ReviewEvent (id, businessId, newReviewCount, avgRating, totalReviews, checkedAt, createdAt)
       VALUES (?, ?, ?, ?, ?, ?, ?)`
    ).run(
      id,
      data.businessId,
      data.newReviewCount,
      data.avgRating,
      data.totalReviews,
      data.checkedAt,
      nowISO()
    );
    return db.prepare('SELECT * FROM ReviewEvent WHERE id = ?').get(id) as unknown as ReviewEvent;
  },
  latest(businessId: string): ReviewEvent | undefined {
    return db
      .prepare(
        'SELECT * FROM ReviewEvent WHERE businessId = ? ORDER BY checkedAt DESC LIMIT 1'
      )
      .get(businessId) as ReviewEvent | undefined;
  },
  listRecent(businessId: string, limit: number): ReviewEvent[] {
    return db
      .prepare(
        'SELECT * FROM ReviewEvent WHERE businessId = ? ORDER BY checkedAt DESC LIMIT ?'
      )
      .all(businessId, limit) as unknown as ReviewEvent[];
  },
};

export default db;
