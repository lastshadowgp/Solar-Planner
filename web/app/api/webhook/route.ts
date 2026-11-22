import type Stripe from 'stripe';
import { db, isProduct, PRODUCTS, stripe, triggerEngine } from '@/lib/server';
export async function POST(req: Request) {
  let event: Stripe.Event;
  try {
    event = stripe.webhooks.constructEvent(await req.text(), req.headers.get('stripe-signature')!, process.env.STRIPE_WEBHOOK_SECRET!);
  } catch { return new Response('bad signature', { status: 400 }); }
  if (event.type === 'checkout.session.completed' || event.type === 'checkout.session.async_payment_succeeded') {
    const s = event.data.object as Stripe.Checkout.Session;
    if (s.payment_status !== 'paid') return new Response('pending');
    const { user_id, product, project_id } = s.metadata ?? {};
    if (!isProduct(product)) return new Response('unknown product');
    if (product === 'single') {
      const { data } = await db.from('projects').update({ status: 'paid', paid_via: 'stripe', updated_at: new Date().toISOString() }).eq('id', project_id).eq('status', 'draft').select('id');
      if (data?.length) await triggerEngine(project_id!);
    } else {
      await db.rpc('add_credits', { p_user: user_id, p_amount: PRODUCTS[product].credits, p_session: s.id });
    }
  }
  return new Response('ok');
}
