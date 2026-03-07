import { useState } from "react";
import { Tree, type TreeNodeInfo } from "@blueprintjs/core";
import { useNavigate, useLocation } from "react-router-dom";

interface NavItem {
  id: string;
  label: string;
  path: string;
  icon?: string;
}

interface NavGroup {
  id: string;
  label: string;
  items: NavItem[];
}

const NAV_GROUPS: NavGroup[] = [
  {
    id: "thrust-chamber",
    label: "Thrust Chamber",
    items: [
      { id: "engine", label: "Engine Design", path: "/engine", icon: "flame" },
      { id: "cooling", label: "Cooling Analysis", path: "/cooling", icon: "snowflake" },
      { id: "contour", label: "Nozzle Contour", path: "/contour", icon: "curved-range-tree" },
    ],
  },
  {
    id: "performance",
    label: "Performance",
    items: [
      { id: "throttle", label: "Throttle Analysis", path: "/throttle", icon: "dashboard" },
      { id: "analysis", label: "Off-Design", path: "/analysis", icon: "function" },
    ],
  },
  {
    id: "statistical",
    label: "Analysis",
    items: [
      { id: "monte-carlo", label: "Monte Carlo", path: "/monte-carlo", icon: "scatter-plot" },
      { id: "optimization", label: "Optimization", path: "/optimization", icon: "trending-up" },
    ],
  },
  {
    id: "components",
    label: "Components",
    items: [
      { id: "injector", label: "Injector Design", path: "/injector", icon: "filter" },
      { id: "igniter", label: "Igniter Design", path: "/igniter", icon: "torch" },
      { id: "tank", label: "Tank Simulation", path: "/tank", icon: "database" },
    ],
  },
  {
    id: "settings-group",
    label: "Settings",
    items: [
      { id: "projects", label: "Projects", path: "/projects", icon: "folder-open" },
      { id: "settings", label: "Settings", path: "/settings", icon: "cog" },
    ],
  },
];

export function NavigationSidebar() {
  const navigate = useNavigate();
  const location = useLocation();
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(
    new Set(NAV_GROUPS.map((g) => g.id))
  );

  function buildNodes(): TreeNodeInfo[] {
    return NAV_GROUPS.map((group) => ({
      id: group.id,
      label: (
        <span
          style={{
            fontSize: "10px",
            fontWeight: 700,
            letterSpacing: "0.1em",
            textTransform: "uppercase",
            color: "#5c7d9e",
          }}
        >
          {group.label}
        </span>
      ),
      isExpanded: expandedGroups.has(group.id),
      hasCaret: true,
      childNodes: group.items.map((item) => ({
        id: item.id,
        label: item.label,
        isSelected: location.pathname === item.path,
        nodeData: item.path,
        icon: item.icon as TreeNodeInfo["icon"],
      })),
    }));
  }

  function handleNodeClick(node: TreeNodeInfo) {
    if (node.childNodes) {
      // Toggle group
      setExpandedGroups((prev) => {
        const next = new Set(prev);
        if (next.has(String(node.id))) {
          next.delete(String(node.id));
        } else {
          next.add(String(node.id));
        }
        return next;
      });
    } else if (node.nodeData) {
      navigate(String(node.nodeData));
    }
  }

  return (
    <div
      style={{
        width: 220,
        minWidth: 220,
        background: "#0a1929",
        borderRight: "1px solid #1e3a5f",
        overflowY: "auto",
        paddingTop: 8,
      }}
    >
      <div
        style={{
          padding: "12px 16px 8px",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <span
          style={{
            fontSize: "15px",
            fontWeight: 800,
            color: "#7ba7cc",
            letterSpacing: "0.05em",
          }}
        >
          RESA
        </span>
        <span style={{ fontSize: "10px", color: "#5c7d9e" }}>v2.0</span>
      </div>
      <Tree
        contents={buildNodes()}
        onNodeClick={handleNodeClick}
        onNodeCollapse={handleNodeClick}
        onNodeExpand={handleNodeClick}
      />
    </div>
  );
}
