import { db, getUser, json } from '@/lib/server';
export async function GET(req: Request) {
  const user = await getUser(req);
  if (!user) return json({ error: 'Unauthorized' }, 401);
  const { data: bal } = await db.from('credit_balances').select('balance').eq('user_id', user.id).maybeSingle();
  const { data: projects } = await db.from('projects').select('id,status,created_at').eq('user_id', user.id).neq('status', 'draft').order('created_at', { ascending: false }).limit(20);
  return json({ email: user.email, balance: bal?.balance ?? 0, projects });
}
