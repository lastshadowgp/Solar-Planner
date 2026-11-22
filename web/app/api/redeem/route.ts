import { db, getUser, json, triggerEngine } from '@/lib/server';
export async function POST(req: Request) {
  const user = await getUser(req);
  if (!user) return json({ error: 'Unauthorized' }, 401);
  const { projectId } = await req.json();
  const { data: ok, error } = await db.rpc('use_credit', { p_user: user.id, p_project: projectId });
  if (error || !ok) return json({ error: 'Insufficient credits' }, 400);
  await triggerEngine(projectId);
  return json({ ok: true });
}
