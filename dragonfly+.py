import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') # Or use 'QtAgg' if you installed PyQt6
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
import argparse
import sys
import math

# --- Default Visualization Parameters ---
# Set your preferred default size and resolution here
DEFAULT_FIG_WIDTH = 50  # Default figure width in inches
DEFAULT_FIG_HEIGHT = 15 # Default figure height in inches
DEFAULT_DPI = 144       # Default dots per inch

class DragonflyBFT:
    """
    Dragonfly Network Generator with Balanced Fat-Tree (BFT) Intra-Group Topology.

    V2 Layout:
    - Groups arranged on a 2g-gon in the XY plane (groups at even vertices).
    - T1s, T2s, Hosts of a group lie in a TANGENTIAL vertical plane.
    - T2 switches lie on the XY plane (z=0), along the tangent line.
    - T1 switches lie parallel to T2s but offset vertically (e.g., z=offset).
    - Hosts lie within the tangential plane, near their T1.
    """

    def __init__(self, radix, total_hosts):
        # (Initialization code remains largely the same as the previous version)
        if radix % 2 != 0:
            raise ValueError(f"Switch Radix (R={radix}) must be even for the R/2 up/down BFT structure.")
        if radix < 4:
             raise ValueError(f"Switch Radix (R={radix}) must be at least 4.")

        self.R = radix
        self.n_t1 = self.R // 2
        self.n_t2 = self.R // 2
        self.p = self.R // 2
        self.h = self.R // 2

        self.hosts_per_group = self.n_t1 * self.p
        if self.hosts_per_group == 0:
             raise ValueError("Calculated hosts per group is zero. Check radix.")

        if total_hosts == 0:
             raise ValueError("Total hosts cannot be zero.")

        # Calculate g, handling non-exact divisibility
        self.g_float = total_hosts / self.hosts_per_group
        self.g = math.ceil(self.g_float) # Use ceiling to ensure all hosts fit

        if self.g == 0: self.g = 1 # Ensure at least one group if hosts > 0

        if self.g_float != self.g:
             print(f"WARNING: Target total hosts ({total_hosts}) requires {self.g_float:.2f} groups.")
             print(f"         Rounding up to g = {self.g} groups.")
             print(f"         The last group will contain fewer hosts/switches than others if not fully utilized.")
             # Future enhancement: Implement partial last group in _create_graph and layout

        self.total_hosts_actual = total_hosts # User's target
        self.total_hosts_structured = self.g * self.hosts_per_group # Hosts capacity of the structure

        self.is_single_group = (self.g == 1)

        # Calculate total components
        self.total_t1_switches = self.g * self.n_t1
        self.total_t2_switches = self.g * self.n_t2
        self.total_switches = self.total_t1_switches + self.total_t2_switches

        # Calculate link counts
        # Adjust host links count based on actual target, not structured capacity
        self.host_links = self.total_hosts_actual
        self.local_t1_t2_links = self.g * self.n_t1 * self.n_t2 # Full bipartite T1<->T2
        self.total_global_endpoints = self.g * self.n_t2 * self.h if not self.is_single_group else 0
        self.global_links_count = self.total_global_endpoints // 2
        self.total_links = self.host_links + self.local_t1_t2_links + self.global_links_count

        # Port Checks
        self.t1_radix_calc = self.p + self.n_t2
        self.t2_radix_calc = self.n_t1 + self.h
        if self.t1_radix_calc > self.R or self.t2_radix_calc > self.R:
             raise ValueError(f"Internal Error: Calculated radix mismatch T1={self.t1_radix_calc}, T2={self.t2_radix_calc} vs R={self.R}")

        # Global links per pair (average)
        if not self.is_single_group and self.g > 1:
             denom = self.g * (self.g - 1)
             self.avg_glp = self.total_global_endpoints / denom if denom > 0 else 0
        else:
             self.avg_glp = 0

        # Create the graph
        self.G = self._create_graph() # Assumes full groups for now

        # Generate positions for visualization
        self.pos_3d = self._generate_3d_positions()

    def _create_graph(self):
        # (This function remains the same as the previous version)
        # It builds the graph topology based on integer 'g' full groups.
        # It includes T1, T2, Hosts and connects them locally.
        # It attempts to create global links between T2 switches.
        # NOTE: If g was rounded up, this creates a full structure for g groups.
        # The host connection part *should* stop adding hosts past total_hosts_actual.

        G = nx.Graph()
        host_count = 0

        for g_idx in range(self.g):
            # Add T1 switches
            for t1_idx in range(self.n_t1):
                t1_id = f"T1-{g_idx}-{t1_idx}"
                G.add_node(t1_id, type='T1', group=g_idx, index=t1_idx)

                # Add Hosts connected to this T1
                for h_idx in range(self.p):
                    if host_count >= self.total_hosts_actual:
                        break # Stop adding hosts once target is reached

                    host_id = f"H-{g_idx}-{t1_idx}-{h_idx}"
                    G.add_node(host_id, type='host', group=g_idx, t1=t1_idx, index=h_idx)
                    G.add_edge(t1_id, host_id, type='host')
                    host_count += 1
                if host_count >= self.total_hosts_actual: break # Break outer loop too
            if host_count >= self.total_hosts_actual and g_idx < self.g -1 :
                 print(f"INFO: Reached target host count ({self.total_hosts_actual}) within group {g_idx}.")
                 # Need to decide if we continue adding switches for remaining groups (for structure)
                 # or stop graph generation early. Let's continue adding switches for layout consistency.
                 pass

            # Add T2 switches (always add switches for the full 'g' groups)
            for t2_idx in range(self.n_t2):
                t2_id = f"T2-{g_idx}-{t2_idx}"
                G.add_node(t2_id, type='T2', group=g_idx, index=t2_idx)

            # Add T1 <-> T2 links (full bipartite for this group)
            for t1_idx in range(self.n_t1):
                 # Check if T1 node actually exists (might not if host limit reached early)
                 t1_id_check = f"T1-{g_idx}-{t1_idx}"
                 if t1_id_check not in G: continue

                 for t2_idx in range(self.n_t2):
                    t1_id = f"T1-{g_idx}-{t1_idx}"
                    t2_id = f"T2-{g_idx}-{t2_idx}"
                    G.add_edge(t1_id, t2_id, type='local')

        # --- Create Global Connections --- (Same logic as before)
        if not self.is_single_group:
            endpoints_per_group = self.n_t2 * self.h
            glp = 0
            if self.g > 1:
                 if endpoints_per_group % (self.g - 1) != 0:
                      print(f"WARNING: Global links per group pair (glp = {endpoints_per_group}/{(self.g - 1)}) is not an integer.")
                      glp = math.ceil(endpoints_per_group / (self.g - 1))
                 else:
                      glp = endpoints_per_group // (self.g - 1)

            actual_global_links_per_t2 = {node_id: 0 for node_id, data in G.nodes(data=True) if data['type'] == 'T2'}
            added_links_count = 0

            for g1 in range(self.g):
                for g2 in range(g1 + 1, self.g):
                    links_added_this_pair = 0
                    # Try to add ~glp links, respecting T2 port limits (h)
                    for k in range(glp * 2): # Iterate more to find available slots
                        if links_added_this_pair >= glp and glp > 0: break # Limit links per pair somewhat

                        s1_idx = k % self.n_t2
                        s2_idx = (k + g1 + g2) % self.n_t2

                        t2_1_id = f"T2-{g1}-{s1_idx}"
                        t2_2_id = f"T2-{g2}-{s2_idx}"

                        # Check if nodes exist and ports are available
                        if (t2_1_id in G and t2_2_id in G and
                            t2_1_id in actual_global_links_per_t2 and actual_global_links_per_t2[t2_1_id] < self.h and
                            t2_2_id in actual_global_links_per_t2 and actual_global_links_per_t2[t2_2_id] < self.h and
                            added_links_count < self.global_links_count and
                            not G.has_edge(t2_1_id, t2_2_id)):

                            G.add_edge(t2_1_id, t2_2_id, type='global')
                            actual_global_links_per_t2[t2_1_id] += 1
                            actual_global_links_per_t2[t2_2_id] += 1
                            added_links_count += 1
                            links_added_this_pair += 1


            # --- Validate Global Connection Distribution ---
            if actual_global_links_per_t2:
                 counts = list(actual_global_links_per_t2.values())
                 min_links = min(counts) if counts else 0
                 max_links = max(counts) if counts else 0
                 print(f"INFO: Global link distribution per T2 switch: Min={min_links}, Max={max_links}. Target Ports={self.h}")
                 # Don't warn strictly, as balancing is hard with simple schemes
                 # if min_links != max_links or max_links > self.h:
                 #      print(f"WARNING: Uneven or potentially incorrect global link count per T2 switch detected!")
                 if added_links_count != self.global_links_count:
                      print(f"WARNING: Number of global links added ({added_links_count}) differs from expected calculation ({self.global_links_count})")

        return G


    def _generate_3d_positions(self):
        """
        Generate 3D positions (V7 Layout):
        - Groups centered on EVEN vertices of a 2g-gon in XY plane.
        - T2 switches on XY plane (Z=0), spread tangentially (wider).
        - T1 switches spread tangentially at Z = -vertical_offset (beneath T2).
        - Hosts form a HORIZONTAL CIRCLE BELOW T1 (further beneath T2).
        - Adjusted intra- and inter-group spacing.
        """
        pos = {}
        if self.g == 0: return pos

        # --- Layout Parameters (ADJUSTED DEFAULTS for V8) ---
        # Increased tangential spread (same as V7)
        internal_scale = max(self.n_t1, self.n_t2, 1) * 1.0
        # INCREASED vertical offset for more group height
        vertical_offset = 0.2 # T1s will be at Z = -1.0
        # Host circle parameters (keep radius small, increase vertical drop)
        host_circle_radius = 0.4 # Radius of the host circle in XY projection
        host_vertical_drop = 0.1 # How far BELOW T1 the host circle's Z-level is (increased)

        # --- Polygon Vertices & Inter-Group Spacing ---
        num_polygon_vertices = max(self.g * 2, 3) # Use 2g-gon
        # Factor relating inter-group center distance to intra-group width
        # Setting k=2.0 means center-to-center distance is ~2x internal_scale
        group_radius_factor_k = 2.0

        # Calculate radius based on keeping groups separated by factor k * width
        # Circumference = num_vertices * desired_separation
        # 2 * pi * group_radius ≈ num_polygon_vertices * (group_radius_factor_k * internal_scale)
        group_radius = (num_polygon_vertices * group_radius_factor_k * internal_scale) / (2 * np.pi)
        # Ensure a minimum radius, especially for small g
        group_radius = max(group_radius, internal_scale * 1.2) # Radius > half-width

        polygon_angles = np.linspace(0, 2*np.pi, num_polygon_vertices, endpoint=False)
        polygon_vertices = {
            v_idx: (group_radius * np.cos(angle), group_radius * np.sin(angle), 0)
            for v_idx, angle in enumerate(polygon_angles)
        }

        # --- Generate Positions for Nodes ---
        for g_idx in range(self.g):
            group_vertex_idx = 2 * g_idx
            if group_vertex_idx >= num_polygon_vertices: group_vertex_idx = 0

            group_cx, group_cy, group_cz = polygon_vertices[group_vertex_idx] # Z=0
            group_angle = polygon_angles[group_vertex_idx]
            cos_a = np.cos(group_angle)
            sin_a = np.sin(group_angle)

            # --- T2 Switch Positions (On XY plane, Z=0) ---
            # Spread wider using updated internal_scale
            if self.n_t2 > 0:
                 t2_tangential_coords = np.linspace(-internal_scale / 2, internal_scale / 2, self.n_t2) \
                                       if self.n_t2 > 1 else np.array([0.0])
            else:
                 t2_tangential_coords = np.array([])
            for t2_idx, t in enumerate(t2_tangential_coords):
                t2_id = f"T2-{g_idx}-{t2_idx}"
                if t2_id not in self.G: continue
                x = group_cx - t * sin_a
                y = group_cy + t * cos_a
                z = 0.0 # T2 at global plane Z=0
                pos[t2_id] = (x, y, z)

            # --- T1 Switch Positions (BELOW T2 plane, Z = -vertical_offset) ---
            # Spread wider using updated internal_scale
            if self.n_t1 > 0:
                 t1_tangential_coords = np.linspace(-internal_scale / 2, internal_scale / 2, self.n_t1) \
                                       if self.n_t1 > 1 else np.array([0.0])
            else:
                 t1_tangential_coords = np.array([])
            for t1_idx, t in enumerate(t1_tangential_coords):
                t1_id = f"T1-{g_idx}-{t1_idx}"
                if t1_id not in self.G: continue

                x = group_cx - t * sin_a
                y = group_cy + t * cos_a
                z = -vertical_offset # T1 BELOW global plane (Negative Z)
                pos[t1_id] = (x, y, z)

                # --- Host Positioning (Horizontal circle BELOW T1) ---
                t1_x, t1_y, t1_z = pos[t1_id] # T1's final position (t1_z is negative)

                connected_hosts = [neighbor for neighbor in self.G.neighbors(t1_id)
                                   if self.G.nodes[neighbor].get('type') == 'host']
                num_hosts_on_t1 = len(connected_hosts)
                if num_hosts_on_t1 == 0: continue

                # Calculate the Z coordinate for the host circle plane (further below T1)
                host_circle_z = t1_z - host_vertical_drop # Subtract drop from already negative t1_z

                # Calculate angles for hosts around the horizontal circle
                host_angles = np.linspace(0, 2*np.pi, num_hosts_on_t1, endpoint=False)

                for h_idx, host_id in enumerate(connected_hosts):
                    if host_id not in self.G: continue

                    angle = host_angles[h_idx]

                    # Calculate host X, Y, Z in a horizontal circle centered at (t1_x, t1_y)
                    # Use smaller radius for shorter host "cables"
                    host_x = t1_x + host_circle_radius * np.cos(angle)
                    host_y = t1_y + host_circle_radius * np.sin(angle)
                    host_z = host_circle_z # All hosts at the same negative Z level

                    pos[host_id] = (host_x, host_y, host_z)

        return pos

    def visualize_static(self, show_hosts=True, show_axes=False,
                         elev=30, azim=30, figsize=(16, 14), # Default figsize
                         node_sizes=None, alpha_links=None, global_lw=1.0,
                         node_labels=False, outfile=None, fig_dpi=100): # Added fig_dpi
        """Visualize the Dragonfly BFT topology in 3D with manual limits."""

        # Use figsize and fig_dpi when creating figure
        fig = plt.figure(figsize=figsize, dpi=fig_dpi)
        ax = fig.add_subplot(111, projection='3d')

        if not show_axes:
            ax.set_axis_off()
        else:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        # Node styles (Using V8 Palette)
        default_node_sizes = {'T1': 60, 'T2': 70, 'host': 15}
        if node_sizes is not None: default_node_sizes.update(node_sizes)
        node_sizes = default_node_sizes
        node_colors = {'T1': 'mediumorchid', 'T2': 'navy', 'host': 'lightcoral'}
        node_labels_map = {'T1': 'T1 Switches', 'T2': 'T2 Switches', 'host': 'Hosts'}

        # Link styles (Using V8 Palette)
        default_alpha_links = {'host': 0.15, 'local': 0.3, 'global': 0.6}
        if alpha_links is not None: default_alpha_links.update(alpha_links)
        alpha_links = default_alpha_links
        link_colors = {'host': 'silver', 'local': 'indigo', 'global': 'crimson'}
        link_lws = {'host': 0.5, 'local': 1.0, 'global': global_lw}

        # --- Prepare Node Data ---
        nodes_data = {ntype: {'pos': [], 'ids': []} for ntype in node_colors}
        valid_nodes = set(self.pos_3d.keys()) & set(self.G.nodes())

        for node in valid_nodes:
             data = self.G.nodes[node]
             node_type = data['type']
             if node_type == 'host' and not show_hosts: continue
             if node_type in nodes_data:
                nodes_data[node_type]['pos'].append(self.pos_3d[node])
                nodes_data[node_type]['ids'].append(node)

        # --- Plot Nodes ---
        handles = []
        plotted_node_ids = set()
        for ntype, data in nodes_data.items():
            if data['pos']:
                xs, ys, zs = zip(*data['pos'])
                scatter = ax.scatter(xs, ys, zs, c=node_colors[ntype], s=node_sizes[ntype],
                                     alpha=0.9, edgecolors='black', linewidth=0.4,
                                     label=node_labels_map.get(ntype, ntype))
                handles.append(scatter)
                plotted_node_ids.update(data['ids'])

        # --- Prepare and Plot Edges using Line3DCollection ---
        edge_segments = {etype: [] for etype in link_colors}
        plotted_edges = set()
        for u, v, data in self.G.edges(data=True):
            if u in plotted_node_ids and v in plotted_node_ids:
                edge_tuple = tuple(sorted((u,v)))
                if edge_tuple in plotted_edges: continue
                edge_type = data['type']
                if edge_type in edge_segments:
                     if edge_type == 'host' and not show_hosts: continue
                     if u not in self.pos_3d or v not in self.pos_3d: continue
                     edge_segments[edge_type].append([self.pos_3d[u], self.pos_3d[v]])
                     plotted_edges.add(edge_tuple)

        for etype, segments in edge_segments.items():
            if segments:
                line_col = Line3DCollection(segments,
                                            colors=link_colors[etype],
                                            linewidths=link_lws[etype],
                                            alpha=alpha_links[etype],
                                            label=f'{etype.capitalize()} Links')
                ax.add_collection(line_col)

        # Add node labels if requested
        if node_labels:
             for node_id in plotted_node_ids:
                  if node_id in self.pos_3d:
                     x, y, z = self.pos_3d[node_id]
                     ax.text(x, y, z, node_id, size=6, zorder=10, color='k', ha='center', va='center')

        # Create title
        title = f"Dragonfly (BFT Groups, Tangential Layout) R={self.R}, Target Hosts={self.total_hosts_actual}\n"
        title += f"g={self.g}, T1/grp={self.n_t1}, T2/grp={self.n_t2}, Hosts/T1={self.p} => {self.total_hosts_structured} Max Hosts\n"
        avg_glp_str = f"{self.avg_glp:.2f}" if not self.is_single_group else "N/A"
        title += f"T2 Global Ports={self.h}, Total Global Links={self.global_links_count}, Avg Links/Pair={avg_glp_str}"
        ax.set_title(title, fontsize=10)

        # Add legend
        if handles:
            ax.legend(handles=handles, loc='upper right', fontsize=8, title="Node Types")

        # <----------------- START: INSERTED CODE ----------------->
        # --- Calculate Bounds and Set Manual Axis Limits ---
        all_plotted_coords = [self.pos_3d[node_id] for node_id in plotted_node_ids if node_id in self.pos_3d]

        if all_plotted_coords:
            min_coords = np.min(all_plotted_coords, axis=0)
            max_coords = np.max(all_plotted_coords, axis=0)
            ranges = np.abs(max_coords - min_coords)

            # Add a margin (e.g., 7% of the range, or a minimum absolute value)
            margin_x = max(ranges[0] * 0.07, 1.0)
            margin_y = max(ranges[1] * 0.07, 1.0)
            margin_z = max(ranges[2] * 0.07, 1.0)

            print(f"Setting plot limits with margins: "
                  f"X=({min_coords[0] - margin_x:.2f}, {max_coords[0] + margin_x:.2f}), "
                  f"Y=({min_coords[1] - margin_y:.2f}, {max_coords[1] + margin_y:.2f}), "
                  f"Z=({min_coords[2] - margin_z:.2f}, {max_coords[2] + margin_z:.2f})")

            ax.set_xlim(min_coords[0] - margin_x, max_coords[0] + margin_x)
            ax.set_ylim(min_coords[1] - margin_y, max_coords[1] + margin_y)
            ax.set_zlim(min_coords[2] - margin_z, max_coords[2] + margin_z)

        # --- Set Aspect Ratio (AFTER setting limits) ---
        # Force cubic aspect ratio - often prevents auto-scaling issues in 3D
        #try:
        #    ax.set_box_aspect([1, 1, 1])
        #    print("Setting box aspect to cubic [1, 1, 1]")
        #except Exception as e:
        #     print(f"Warning: Failed to set cubic box aspect. {e}")
        #     # Fallback if aspect setting fails
        #     pass
        # <----------------- END: INSERTED CODE ----------------->
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.1, hspace=0.1)


        try:
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            print("Attempted to maximize window using Qt.") # Add print for confirmation
        except Exception as e:
            print(f"Could not automatically maximize using Qt: {e}")

        # --- Set View and Layout ---
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout() # Keep this - adjusts spacing for titles etc. within the FIGURE

        # --- Save or Show ---
        if outfile:
             # Use fig_dpi when saving, added bbox_inches='tight'
             plt.savefig(outfile, dpi=fig_dpi, bbox_inches='tight')
             print(f"Saved visualization to {outfile}")
        else:
             plt.show()

        return fig, ax
    
    def visualize_static_2d(self, show_hosts=True, figsize=(12, 12), # Good default size for 2D
                            layout_prog='spring', # 'spring', 'kamada_kawai', 'spectral'
                            node_sizes=None, alpha_links=None, global_lw=0.5,
                            node_labels=False, outfile=None, fig_dpi=300): # Use higher DPI
        """Visualize the Dragonfly BFT topology in 2D."""
        print(f"Generating 2D visualization using {layout_prog} layout...")

        # --- 1. Create a NEW Figure ---
        # This ensures it pops up in a separate window
        fig2, ax2 = plt.subplots(figsize=figsize, dpi=fig_dpi)
        ax2.set_axis_off() # Clean look, remove axes
        ax2.set_title(f"Dragonfly+ 2D Layout (R={self.R}, Hosts={self.total_hosts_actual}, g={self.g})", fontsize=12)
        ax2.set_aspect('equal') # Maintain aspect ratio from layout

        # --- 2. Calculate 2D Layout ---
        # Choose a layout algorithm. Spring is common, Kamada-Kawai often looks good.
        # Spectral might be faster for very large graphs.
        # You might need to experiment. Kamada-Kawai can be slow.
        if layout_prog == 'kamada_kawai':
            # Kamada-Kawai needs a connected graph if components exist
            # If graph might be disconnected (e.g., single group no global), handle it
            if self.is_single_group or not nx.is_connected(self.G):
                print("Warning: Graph not fully connected, Kamada-Kawai might behave unexpectedly or fail. Consider spring/spectral.")
                # Simple fallback or draw components separately (more complex)
                pos_2d = nx.spring_layout(self.G, seed=42)
            else:
                try:
                    pos_2d = nx.kamada_kawai_layout(self.G)
                except Exception as e:
                    print(f"Warning: kamada_kawai failed ({e}), falling back to spring layout.")
                    pos_2d = nx.spring_layout(self.G, seed=42)

        elif layout_prog == 'spectral':
            pos_2d = nx.spectral_layout(self.G)
        else: # Default to spring
            pos_2d = nx.spring_layout(self.G, seed=42) # Use a seed for reproducible layouts

        # --- 3. Prepare Nodes and Edges for Drawing ---
        # Re-use color/size logic, potentially adjusting sizes for 2D clarity
        default_node_sizes = {'T1': 35, 'T2': 45, 'host': 5} # Smaller defaults for 2D
        if node_sizes is not None: default_node_sizes.update(node_sizes)
        node_sizes = default_node_sizes
        node_colors = {'T1': 'mediumorchid', 'T2': 'navy', 'host': 'lightcoral'}

        default_alpha_links = {'host': 0.05, 'local': 0.15, 'global': 0.25} # Lower alpha for 2D
        if alpha_links is not None: default_alpha_links.update(alpha_links)
        alpha_links = default_alpha_links
        link_colors = {'host': 'silver', 'local': 'indigo', 'global': 'crimson'}
        link_lws = {'host': 0.3, 'local': 0.6, 'global': global_lw}

        nodes_to_draw = {ntype: [] for ntype in node_colors}
        edges_to_draw = {etype: [] for etype in link_colors}

        valid_nodes_in_pos = set(self.G.nodes()) & set(pos_2d.keys())

        for node in valid_nodes_in_pos:
            node_type = self.G.nodes[node]['type']
            if node_type == 'host' and not show_hosts: continue
            if node_type in nodes_to_draw:
                nodes_to_draw[node_type].append(node)

        for u, v, data in self.G.edges(data=True):
             # Ensure both nodes have positions and are of types we draw
             if u in valid_nodes_in_pos and v in valid_nodes_in_pos:
                edge_type = data['type']
                u_type = self.G.nodes[u]['type']
                v_type = self.G.nodes[v]['type']

                # Skip host edges if hosts are hidden
                if edge_type == 'host' and not show_hosts: continue
                # Skip edges if either node type is hidden (relevant if only showing switches)
                if u_type == 'host' and not show_hosts: continue
                if v_type == 'host' and not show_hosts: continue

                if edge_type in edges_to_draw:
                    edges_to_draw[edge_type].append((u,v))


        # --- 4. Draw Nodes and Edges (Layered) ---
        # Draw edges first, then nodes on top
        for etype, edges in edges_to_draw.items():
             if edges:
                  nx.draw_networkx_edges(self.G, pos_2d, ax=ax2,
                                         edgelist=edges,
                                         edge_color=link_colors[etype],
                                         width=link_lws[etype],
                                         alpha=alpha_links[etype])

        handle_map = {} # For creating legend handles manually
        for ntype, nodes in nodes_to_draw.items():
             if nodes:
                  nx.draw_networkx_nodes(self.G, pos_2d, ax=ax2,
                                         nodelist=nodes,
                                         node_color=node_colors[ntype],
                                         node_size=node_sizes[ntype],
                                         label=ntype) # Simple label for handle creation
                  # Create a proxy artist for the legend
                  handle_map[ntype] = plt.Line2D([0], [0], marker='o', color='w', # White dummy line
                                          markerfacecolor=node_colors[ntype],
                                          markersize=np.sqrt(node_sizes[ntype]), # Approx size match
                                          label=f"{ntype} Switches" if ntype != 'host' else "Hosts")


        # --- 5. Add Legend ---
        if handle_map:
             # Sort handles (e.g., T2, T1, Host)
             sorted_handles = [handle_map[k] for k in ['T2','T1','host'] if k in handle_map]
             ax2.legend(handles=sorted_handles, loc='best', fontsize=8)

        # --- 6. Node Labels (Optional, likely too cluttered) ---
        if node_labels:
             nx.draw_networkx_labels(self.G, pos_2d, ax=ax2, font_size=4)

        # --- 7. Adjust Layout and Save/Show ---
        # No tight_layout needed as we turned axes off, use subplots_adjust if needed
        # fig2.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95) # Example

        # Return the figure and axes objects
        return fig2, ax2

    def print_topology_info(self):
        # (Remains the same as previous version)
        print("\n=== Dragonfly (BFT Groups, Tangential Layout) Topology Information ===")
        print(f"Inputs: Radix R={self.R}, Target Total Hosts={self.total_hosts_actual}")
        print("-" * 20)
        print(f"Derived Structure:")
        print(f"  Groups (g): {self.g}")
        print(f"  T1 Switches per Group (n_t1 = R/2): {self.n_t1}")
        print(f"  T2 Switches per Group (n_t2 = R/2): {self.n_t2}")
        print(f"  Hosts per T1 Switch (p = R/2): {self.p}")
        print(f"  Hosts per Group (n_t1 * p): {self.hosts_per_group}")
        print(f"  Global Ports per T2 Switch (h = R/2): {self.h}")
        print("-" * 20)
        print(f"Network Size:")
        print(f"  Target Hosts: {self.total_hosts_actual}")
        print(f"  Max Hosts in Structure (g * hosts_per_group): {self.total_hosts_structured}")
        print(f"  Total T1 Switches: {self.total_t1_switches}")
        print(f"  Total T2 Switches: {self.total_t2_switches}")
        print(f"  Total Switches: {self.total_switches}")
        print("-" * 20)
        print(f"Switch Radix Usage (Max = R = {self.R}):")
        print(f"  T1 Radix Used = p + n_t2 = {self.p} + {self.n_t2} = {self.t1_radix_calc}")
        print(f"  T2 Radix Used = n_t1 + h = {self.n_t1} + {self.h} = {self.t2_radix_calc}")
        print("-" * 20)
        print(f"Link Counts:")
        print(f"  Host Links (actual connected): {len([e for e in self.G.edges(data=True) if e[2]['type']=='host'])}")
        print(f"  Local T1<->T2 Links: {len([e for e in self.G.edges(data=True) if e[2]['type']=='local'])}")
        print(f"  Global T2<->T2 Links: {len([e for e in self.G.edges(data=True) if e[2]['type']=='global'])}")
        # Total links calculation might be slightly off if groups are partial
        # print(f"  Total Links: {self.total_links}")
        print("-" * 20)
        if not self.is_single_group:
             print(f"Global Connectivity:")
             print(f"  Total Global Endpoints Configured: {self.total_global_endpoints}")
             avg_glp_str = f"{self.avg_glp:.2f}" if self.avg_glp else "N/A"
             print(f"  Avg Global Links per Group Pair: {avg_glp_str}")
        print("=================================================")

def main():
    # (Argparse and main execution logic remains the same)
    parser = argparse.ArgumentParser(description="Generate and visualize a Dragonfly network with BFT groups (Tangential Layout V2).")
    parser.add_argument('--radix', type=int, required=True, help="Switch radix (must be even, >= 4).")
    parser.add_argument('--total-hosts', type=int, required=True, help="Target total number of hosts.")
    parser.add_argument('--hide-hosts', action='store_true', help="Do not draw host nodes and links.")
    parser.add_argument('--no-vis', action='store_true', help="Do not show the visualization window.")
    parser.add_argument('--outfile', type=str, default=None, help="Save visualization to file (e.g., 'dragonfly_bft_v2.png').")
    parser.add_argument('--elev', type=float, default=25, help="Elevation angle for 3D view.")
    parser.add_argument('--azim', type=float, default=-45, help="Azimuth angle for 3D view.")
    parser.add_argument('--global-lw', type=float, default=0.7, help="Linewidth for global links.")
    parser.add_argument('--global-alpha', type=float, default=0.4, help="Alpha (transparency) for global links.")
    parser.add_argument('--local-alpha', type=float, default=0.2, help="Alpha (transparency) for local T1-T2 links.")
    parser.add_argument('--host-alpha', type=float, default=0.1, help="Alpha (transparency) for host links.")
    parser.add_argument('--node-labels', action='store_true', help="Show node ID labels (can be very cluttered).")
    parser.add_argument('--layout2d', type=str, default='spring', choices=['spring', 'kamada_kawai', 'spectral'],
                        help="NetworkX layout for 2D plot (spring, kamada_kawai, spectral).")
    parser.add_argument('--no-vis2d', action='store_true', help="Do not show the 2D visualization window.")
    parser.add_argument('--outfile2d', type=str, default=None, help="Save 2D visualization to file (e.g., 'dragonfly_2d.png').")

    args = parser.parse_args()

    # Add basic validation for total_hosts > 0
    if args.total_hosts <= 0:
        print("Error: --total-hosts must be greater than 0.", file=sys.stderr)
        sys.exit(1)

    try:
        print("Generating Dragonfly BFT topology (Tangential Layout)...")
        topology = DragonflyBFT(radix=args.radix, total_hosts=args.total_hosts)
        topology.print_topology_info()

        figures_to_show = [] # Keep track of figures created

        if not args.no_vis:
            print("Generating 3D visualization...")
            alpha_links_3d = { # Keep separate alpha settings if needed
                'host': args.host_alpha,
                'local': args.local_alpha,
                'global': args.global_alpha
            }
            fig1, ax1 = topology.visualize_static( # Pass figsize/dpi defaults or from args if added
                show_hosts=not args.hide_hosts,
                elev=args.elev,
                azim=args.azim,
                alpha_links=alpha_links_3d,
                global_lw=args.global_lw,
                node_labels=args.node_labels,
                outfile=args.outfile, # Save 3D plot if specified
                fig_dpi=144 # Example DPI for 3D
            )
            if not args.outfile: # Only add to show list if not saving to file only
                figures_to_show.append(fig1)

        if not args.no_vis2d:
             print("Generating 2D visualization...")
             # Use potentially different alpha values optimized for 2D
             alpha_links_2d = {
                'host': max(args.host_alpha * 0.5, 0.02), # Make host links very faint in 2D
                'local': max(args.local_alpha * 0.8, 0.05),
                'global': max(args.global_alpha * 0.8, 0.1)
             }
             fig2, ax2 = topology.visualize_static_2d(
                 show_hosts=not args.hide_hosts,
                 layout_prog=args.layout2d,
                 alpha_links=alpha_links_2d,
                 global_lw=max(args.global_lw * 0.7, 0.3), # Thinner lines for 2D
                 node_labels=args.node_labels,
                 outfile=args.outfile2d, # Save 2D plot if specified
                 fig_dpi=300 # Higher DPI for 2D plot
             )
             if not args.outfile2d: # Only add to show list if not saving to file only
                figures_to_show.append(fig2)
        else:
             print("Visualization skipped (--no-vis).")

        # --- Call plt.show() ONCE after creating all figures ---
        if figures_to_show:
            print("Displaying figures...")
            plt.show()
        elif not args.no_vis and args.outfile:
             print(f"3D visualization saved to {args.outfile}")
        elif not args.no_vis2d and args.outfile2d:
             print(f"2D visualization saved to {args.outfile2d}")
        elif args.outfile and args.outfile2d:
             print(f"Visualizations saved to {args.outfile} and {args.outfile2d}")
        else:
            print("No visualization generated or shown.")
            

    except ValueError as e:
        print(f"\nError creating topology: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError:
         print("\nError: Missing required libraries (numpy, matplotlib, networkx).", file=sys.stderr)
         print("Please install them (e.g., 'pip install numpy matplotlib networkx')", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        sys.exit(1)

if __name__ == "__main__":
    main()
