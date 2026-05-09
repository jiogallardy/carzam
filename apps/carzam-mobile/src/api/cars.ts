import { api } from "./client";
import type { OwnedCar, OwnedCarCreate } from "./types";

export async function listCars(): Promise<OwnedCar[]> {
  const { data } = await api.get<OwnedCar[]>("/me/cars");
  return data;
}

export async function addCar(body: OwnedCarCreate): Promise<OwnedCar> {
  const { data } = await api.post<OwnedCar>("/me/cars", body);
  return data;
}

export async function deleteCar(id: string): Promise<void> {
  await api.delete(`/me/cars/${id}`);
}
