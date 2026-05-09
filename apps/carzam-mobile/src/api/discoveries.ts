import { api } from "./client";
import type { CarClass, Discovery, Stats } from "./types";

export async function listDiscoveries(): Promise<Discovery[]> {
  const { data } = await api.get<Discovery[]>("/me/discoveries");
  return data;
}

export async function listCarClasses(): Promise<CarClass[]> {
  const { data } = await api.get<CarClass[]>("/car-classes");
  return data;
}

export async function fetchStats(): Promise<Stats> {
  const { data } = await api.get<Stats>("/me/stats");
  return data;
}
