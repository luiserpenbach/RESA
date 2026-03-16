import { Icon } from "@blueprintjs/core";
import { useLocation } from "react-router-dom";
import { useUiStore } from "../../store/uiStore";
import { MODULE_DOCS } from "../../data/moduleDocs";

/**
 * Collapsible bottom panel showing key equations and references for the active module.
 * Sits between the main content row and the status bar in the 4-row grid.
 */
export function MethodsPanel() {
  const { methodsPanelOpen, toggleMethodsPanel } = useUiStore();
  const location = useLocation();

  const doc = MODULE_DOCS[location.pathname];

  if (!doc) return null;

  return (
    <div className={`app-methods-panel ${methodsPanelOpen ? "open" : ""}`}>
      {/* Header row */}
      <div className="methods-panel-header">
        <Icon icon="function" size={11} color="var(--accent-bright)" />
        <span className="methods-panel-title">{doc.title} — Methods &amp; References</span>
        <div className="methods-panel-spacer" />
        {doc.reference && (
          <span className="methods-panel-ref">
            <Icon icon="book" size={10} />
            {doc.reference}
          </span>
        )}
        <button className="methods-panel-close icon-btn" onClick={toggleMethodsPanel} title="Close methods panel">
          <Icon icon="cross" size={11} />
        </button>
      </div>

      {/* Section columns */}
      <div className="methods-panel-body">
        {doc.sections.map((section) => (
          <div key={section.title} className="methods-section">
            <div className="methods-section-title">{section.title}</div>
            <ul className="methods-section-lines">
              {section.lines.map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
