import { db, getUser, isProduct, json, PRODUCTS, SITE, stripe } from '@/lib/server';
export async function POST(req: Request) {
  const user = await getUser(req);
  if (!user) return json({ error: 'Unauthorized' }, 401);
  const { product, projectId } = await req.json();
  if (!isProduct(product)) return json({ error: 'Invalid product' }, 400);
  const item = PRODUCTS[product];
  const session = await stripe.checkout.sessions.create({
    mode: 'payment',
    payment_method_types: ['card', 'promptpay'],
    customer_email: user.email,
    line_items: [{ quantity: 1, price_data: { currency: 'thb', unit_amount: item.amount, product_data: { name: item.name } } }],
    metadata: { user_id: user.id, product, project_id: projectId ?? '' },
    success_url: product === 'single' ? `${SITE}/report.html?id=${projectId}` : `${SITE}/?credits=ok`,
cancel_url: `${SITE}/`,
  });
  return json({ url: session.url });
}
