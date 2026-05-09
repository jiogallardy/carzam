import { api } from "./client";
import type { AuthResponse } from "./types";

export async function signInWithGoogle(idToken: string): Promise<AuthResponse> {
  const { data } = await api.post<AuthResponse>("/auth/google", { id_token: idToken });
  return data;
}

export async function signInWithApple(args: {
  idToken: string;
  authorizationCode?: string | null;
  fullName?: string | null;
  email?: string | null;
}): Promise<AuthResponse> {
  const { data } = await api.post<AuthResponse>("/auth/apple", {
    id_token: args.idToken,
    authorization_code: args.authorizationCode ?? null,
    full_name: args.fullName ?? null,
    email: args.email ?? null,
  });
  return data;
}
