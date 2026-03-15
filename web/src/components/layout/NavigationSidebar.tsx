import { useState } from "react";
import { Tree, type TreeNodeInfo } from "@blueprintjs/core";
import { useNavigate, useLocation } from "react-router-dom";
import { useDesignSessionStore } from "../../store/designSessionStore";
import type { ModuleName, ModuleStatus } from "../../types/session";

interface NavItem {
  id: string;
  label: string;
  path: string;
  icon?: string;
  moduleKey?: ModuleName;
}

interface NavGroup {
  id: string;
  label: string;
  items: NavItem[];
}

const NAV_GROUPS: NavGroup[] = [
  {
    id: "design-workflow",
    label: "Design Workflow",
    items: [
      { id: "engine", label: "Engine Design", path: "/engine", icon: "flame", moduleKey: "engine" },
      { id: "contour", label: "Nozzle Contour", path: "/contour", icon: "curved-range-tree", moduleKey: "contour" },
      { id: "cooling", label: "Cooling Design", path: "/cooling", icon: "snowflake", moduleKey: "cooling" },
      { id: "structural", label: "Wall Thickness", path: "/structural", icon: "shield", moduleKey: "wall_thickness" },
    ],
  },
  {
    id: "performance",
    label: "Performance",
    items: [
      { id: "performance", label: "Performance Maps", path: "/performance", icon: "dashboard", moduleKey: "performance" },
      { id: "feed-system", label: "Feed System", path: "/feed-system", icon: "flow-branch", moduleKey: "feed_system" },
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

const STATUS_COLORS: Record<ModuleStatus, string> = {
  completed: "#43a047",
  ready: "#5c7d9e",
  locked: "#2a3f54",
  stale: "#e65100",
};

function StatusDot({ status }: { status: ModuleStatus }) {
  return (
    <span
      style={{
        display: "inline-block",
        width: 6,
        height: 6,
        borderRadius: "50%",
        backgroundColor: STATUS_COLORS[status],
        marginLeft: 6,
        flexShrink: 0,
      }}
    />
  );
}

export function NavigationSidebar() {
  const navigate = useNavigate();
  const location = useLocation();
  const moduleStatus = useDesignSessionStore((s) => s.moduleStatus);
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
      childNodes: group.items.map((item) => {
        const status = item.moduleKey ? moduleStatus[item.moduleKey] : undefined;
        return {
          id: item.id,
          label: (
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              {item.label}
              {status && <StatusDot status={status} />}
            </span>
          ),
          isSelected: location.pathname === item.path,
          nodeData: item.path,
          icon: item.icon as TreeNodeInfo["icon"],
        };
      }),
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
