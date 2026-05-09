import { api } from "./client";
import type { ClipOut } from "./types";

export async function uploadClipForCar(carId: string, uri: string): Promise<ClipOut> {
  const form = new FormData();
  // @ts-expect-error — RN-specific FormData entry shape
  form.append("audio", { uri, name: "clip.wav", type: "audio/wav" });
  const { data } = await api.post<ClipOut>(`/me/cars/${carId}/clips`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}
