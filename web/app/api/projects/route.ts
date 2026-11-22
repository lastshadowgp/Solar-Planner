import { db, getUser, json, validateProject } from '@/lib/server';
export async function POST(req: Request) {
  const user = await getUser(req);
  if (!user) return json({ error: 'Unauthorized' }, 401);
  const data = await req.json();
  const err = validateProject(data);
  if (err) return json({ error: err }, 400);
  const { data: p, error } = await db.from('projects').insert({ user_id: user.id, data }).select('id').single();
  if (error) return json({ error: error.message }, 500);
  return json({ id: p.id });
}
