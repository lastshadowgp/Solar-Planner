create table public.projects (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users on delete set null,
  data jsonb not null,
  status text not null default 'draft' check (status in ('draft','paid','processing','done','error')),
  paid_via text,
  result jsonb,
  pdf_path text,
  error text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table public.credit_balances (
  user_id uuid primary key references auth.users on delete cascade,
  balance int not null default 0 check (balance >= 0)
);

create table public.credit_ledger (
  id bigserial primary key,
  user_id uuid not null references auth.users on delete cascade,
  delta int not null,
  reason text not null,
  stripe_session_id text unique,
  project_id uuid references public.projects,
  created_at timestamptz default now()
);

alter table projects enable row level security;
alter table credit_balances enable row level security;
alter table credit_ledger enable row level security;

create or replace function public.add_credits(p_user uuid, p_amount int, p_session text)
returns void language plpgsql security definer set search_path = public as 
begin
  insert into credit_ledger(user_id, delta, reason, stripe_session_id) values (p_user, p_amount, 'purchase', p_session);
  insert into credit_balances(user_id, balance) values (p_user, p_amount)
  on conflict (user_id) do update set balance = credit_balances.balance + excluded.balance;
exception when unique_violation then null;
end ;

create or replace function public.use_credit(p_user uuid, p_project uuid)
returns boolean language plpgsql security definer set search_path = public as 
begin
  update credit_balances set balance = balance - 1 where user_id = p_user and balance > 0;
  if not found then return false; end if;
  update projects set status = 'paid', paid_via = 'credit', updated_at = now() where id = p_project and user_id = p_user and status = 'draft';
  if not found then raise exception 'not found'; end if;
  insert into credit_ledger(user_id, delta, reason, project_id) values (p_user, -1, 'report', p_project);
  return true;
end ;
