import { useState } from "react";
import { Icon, Tree, type TreeNodeInfo } from "@blueprintjs/core";
import { useNavigate, useLocation } from "react-router-dom";
import { useDesignSessionStore } from "../../store/designSessionStore";
import { useUiStore } from "../../store/uiStore";
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
      {
        id: "contour",
        label: "Nozzle Contour",
        path: "/contour",
        icon: "curved-range-tree",
        moduleKey: "contour",
      },
      {
        id: "cooling",
        label: "Cooling Design",
        path: "/cooling",
        icon: "snowflake",
        moduleKey: "cooling",
      },
      {
        id: "structural",
        label: "Wall Thickness",
        path: "/structural",
        icon: "shield",
        moduleKey: "wall_thickness",
      },
    ],
  },
  {
    id: "performance",
    label: "Performance",
    items: [
      {
        id: "performance",
        label: "Performance Maps",
        path: "/performance",
        icon: "dashboard",
        moduleKey: "performance",
      },
      {
        id: "feed-system",
        label: "Feed System",
        path: "/feed-system",
        icon: "flow-branch",
        moduleKey: "feed_system",
      },
    ],
  },
  {
    id: "statistical",
    label: "Analysis",
    items: [
      { id: "monte-carlo", label: "Monte Carlo", path: "/monte-carlo", icon: "scatter-plot", moduleKey: "monte_carlo" },
      { id: "optimization", label: "Optimization", path: "/optimization", icon: "trending-up", moduleKey: "optimization" },
    ],
  },
  {
    id: "components",
    label: "Components",
    items: [
      { id: "injector", label: "Injector Design", path: "/injector", icon: "filter", moduleKey: "injector" },
      { id: "igniter", label: "Igniter Design", path: "/igniter", icon: "flash", moduleKey: "igniter" },
      { id: "tank", label: "Tank Simulation", path: "/tank", icon: "database", moduleKey: "tank" },
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

interface NavigationSidebarProps {
  collapsed?: boolean;
}

export function NavigationSidebar({ collapsed }: NavigationSidebarProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const moduleStatus = useDesignSessionStore((s) => s.moduleStatus);
  const toggleNav = useUiStore((s) => s.toggleNav);
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

  if (collapsed) {
    return (
      <div className="app-nav collapsed">
        <button className="nav-toggle" onClick={toggleNav} title="Expand navigation">
          <Icon icon="menu" size={16} />
        </button>
      </div>
    );
  }

  return (
    <div className="app-nav">
      <div className="nav-header">
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: "var(--accent-bright)",
            }}
          />
          <span
            style={{
              fontSize: "14px",
              fontWeight: 700,
              color: "var(--accent-bright)",
              letterSpacing: "0.08em",
            }}
          >
            RESA
          </span>
          <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>v2.0</span>
        </div>
        <button className="nav-toggle" onClick={toggleNav} title="Collapse navigation">
          <Icon icon="chevron-left" size={14} />
        </button>
      </div>
      <div style={{ flex: 1, overflowY: "auto", paddingTop: 4 }}>
        <Tree
          contents={buildNodes()}
          onNodeClick={handleNodeClick}
          onNodeCollapse={handleNodeClick}
          onNodeExpand={handleNodeClick}
        />
      </div>
    </div>
  );
}
