import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

/** Root route — immediately redirect to the engine workspace. */
export default function HomePage() {
  const navigate = useNavigate();
  useEffect(() => {
    navigate("/engine", { replace: true });
  }, [navigate]);
  return null;
}
