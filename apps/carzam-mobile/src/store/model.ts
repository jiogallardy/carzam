import * as SecureStore from "expo-secure-store";
import { create } from "zustand";

const KEY = "carzam.activeModel";

type ModelState = {
  activeModelId: string | null;
  hydrated: boolean;
  hydrate: () => Promise<void>;
  setActiveModelId: (id: string | null) => Promise<void>;
};

// Holds the user's chosen model id. `null` = let the server pick its default.
export const useModelStore = create<ModelState>((set) => ({
  activeModelId: null,
  hydrated: false,
  hydrate: async () => {
    const v = await SecureStore.getItemAsync(KEY);
    set({ activeModelId: v && v.length > 0 ? v : null, hydrated: true });
  },
  setActiveModelId: async (id) => {
    if (id) await SecureStore.setItemAsync(KEY, id);
    else await SecureStore.deleteItemAsync(KEY);
    set({ activeModelId: id });
  },
}));
