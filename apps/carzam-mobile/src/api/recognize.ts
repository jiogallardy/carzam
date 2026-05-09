import { api } from "./client";
import type { RecognizeResponse, SubmitNewCarRequest } from "./types";

export async function recognizeAudio(uri: string): Promise<RecognizeResponse> {
  const form = new FormData();
  // RN supports the {uri, name, type} shape on FormData entries
  // @ts-expect-error — RN-specific FormData shape
  form.append("audio", { uri, name: "clip.wav", type: "audio/wav" });
  const { data } = await api.post<RecognizeResponse>("/recognize", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function submitAsNewCar(clipId: string, body: SubmitNewCarRequest) {
  const { data } = await api.post(`/clips/${clipId}/submit-as-new-car`, body);
  return data;
}
