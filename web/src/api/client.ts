import axios from "axios";

const baseURL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "/api/v1";

const client = axios.create({
  baseURL,
  headers: {
    "Content-Type": "application/json",
  },
});

export default client;
