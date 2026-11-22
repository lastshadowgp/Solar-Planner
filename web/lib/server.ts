import Stripe from 'stripe';
import { createClient } from '@supabase/supabase-js';

export const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!);
export const db = createClient(process.env.SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!, { auth: { persistSession: false } });
export const SITE = process.env.SITE_URL ?? process.env.RENDER_EXTERNAL_URL!;
const ENGINE_URL = process.env.ENGINE_URL ?? http://;

export const PRODUCTS = {
  single: { name: 'Report 1', amount: 9900, credits: 0 },
  pack10: { name: '10 Credits', amount: 79000, credits: 10 },
  pack30: { name: '30 Credits', amount: 199000, credits: 30 },
} as const;
export type ProductId = keyof typeof PRODUCTS;
export const isProduct = (p: unknown): p is ProductId => typeof p === 'string' && Object.hasOwn(PRODUCTS, p);

export const json = (body: unknown, status = 200) => Response.json(body, { status });

export async function getUser(req: Request) {
  const token = req.headers.get('authorization')?.replace(/^Bearer /, '');
  if (!token) return null;
  const { data } = await db.auth.getUser(token);
  return data.user ?? null;
}

export function triggerEngine(projectId: string) {
  return fetch(${ENGINE_URL}/report, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Engine-Key': process.env.ENGINE_KEY! },
    body: JSON.stringify({ project_id: projectId }),
  }).catch(e => console.error('engine trigger failed', e));
}

export function validateProject(p: any): string | null {
  const num = (v: any) => typeof v === 'number' && Number.isFinite(v);
  if (!num(p?.lat) || Math.abs(p.lat) > 90 || !num(p?.lon) || Math.abs(p.lon) > 180) return 'Invalid lat/lon';
  if (!Array.isArray(p.panels) || p.panels.length < 1 || p.panels.length > 1000) return 'Invalid panels';
  return null;
}
