def make_person_item_map(theta, difficulty, item_names):
    """
    Improved person-item map with non-overlapping labels.
    """

    fig_height = max(7, 0.35 * len(item_names))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    rng = np.random.default_rng(123)

    # Person jitter
    x_person = rng.normal(0, 0.035, len(theta))

    ax.scatter(
        x_person,
        theta,
        alpha=0.25,
        s=14,
        label="Persons",
        color="steelblue"
    )

    ax.scatter(
        np.ones(len(difficulty)),
        difficulty,
        marker="D",
        s=70,
        label="Items",
        color="darkred",
        zorder=3
    )

    # Sort items by difficulty
    item_data = sorted(
        zip(difficulty, item_names),
        key=lambda x: x[0]
    )

    sorted_difficulty = np.array([x[0] for x in item_data])
    sorted_names = [x[1] for x in item_data]

    # Prevent overlap
    min_gap = 0.20
    adjusted_y = sorted_difficulty.copy()

    for i in range(1, len(adjusted_y)):
        if adjusted_y[i] - adjusted_y[i - 1] < min_gap:
            adjusted_y[i] = adjusted_y[i - 1] + min_gap

    # Draw connector lines + labels
    for original_y, label_y, label in zip(
        sorted_difficulty,
        adjusted_y,
        sorted_names
    ):

        ax.plot(
            [1.02, 1.12],
            [original_y, label_y],
            color="gray",
            linewidth=0.8
        )

        ax.text(
            1.14,
            label_y,
            str(label),
            va="center",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.9
            )
        )

    ax.set_xlim(-0.25, 2.2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        "Persons\n(ability)",
        "Items\n(difficulty)"
    ])

    ax.set_ylabel("Logit Scale")

    ax.set_title(
        "Person-Item Map (Non-Overlapping)",
        fontsize=14,
        fontweight="bold"
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        color="black",
        alpha=0.5
    )

    ax.grid(axis="y", alpha=0.25)

    ax.legend(loc="upper left")

    plt.tight_layout()

    return fig
