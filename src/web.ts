import { app } from "./server.js";

const PORT = parseInt(process.env.PORT ?? "8500");

app.listen(PORT, "0.0.0.0", () => {
  console.log(`RAG agent web server running on http://0.0.0.0:${PORT}`);
});
