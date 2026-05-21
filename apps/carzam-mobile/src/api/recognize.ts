import { api } from "./client";
import type {
  AggregateRecognizeResponse,
  CarClass,
  ModelsResponse,
  RecognizeResponse,
  SubmitNewCarRequest,
} from "./types";

export async function recognizeAudio(
  uri: string,
  modelId?: string | null,
): Promise<RecognizeResponse> {
  const form = new FormData();
  // RN supports the {uri, name, type} shape on FormData entries
  // @ts-expect-error — RN-specific FormData shape
  form.append("audio", { uri, name: "clip.wav", type: "audio/wav" });
  if (modelId) form.append("model", modelId);
  const { data } = await api.post<RecognizeResponse>("/recognize", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/**
 * Shazam-style aggregate inference. Send a longer clip (~10-15s) and the
 * server slides 5s windows at 2.5s hop, averaging softmax. Returns
 * `is_confident` + `couldnt_place` so the client can pick its commit
 * strategy.
 *
 * Pass `commit: true` exactly once — that's the call that saves the clip
 * to R2 and updates the user's discovery. All polling calls (if any)
 * should use commit=false.
 */
export async function recognizeAudioAggregate(
  uri: string,
  opts: { modelId?: string | null; commit?: boolean } = {},
): Promise<AggregateRecognizeResponse> {
  const form = new FormData();
  // @ts-expect-error — RN-specific FormData shape
  form.append("audio", { uri, name: "clip.wav", type: "audio/wav" });
  if (opts.modelId) form.append("model", opts.modelId);
  form.append("commit", opts.commit ? "true" : "false");
  const { data } = await api.post<AggregateRecognizeResponse>(
    "/recognize/aggregate",
    form,
    { headers: { "Content-Type": "multipart/form-data" } },
  );
  return data;
}

export async function listModels(): Promise<ModelsResponse> {
  const { data } = await api.get<ModelsResponse>("/models");
  return data;
}

/** Return the car classes a specific model can identify. */
export async function listModelClasses(modelId: string): Promise<CarClass[]> {
  const { data } = await api.get<CarClass[]>(`/models/${modelId}/classes`);
  return data;
}

export async function submitAsNewCar(clipId: string, body: SubmitNewCarRequest) {
  const { data } = await api.post(`/clips/${clipId}/submit-as-new-car`, body);
  return data;
}
