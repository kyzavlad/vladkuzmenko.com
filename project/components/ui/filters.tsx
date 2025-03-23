"use client";

import * as React from "react";
import { Check, Circle, AlertCircle, CheckCircle, CircleDashed, CircleDotDashed, CircleEllipsis, XCircle, SignalHigh, SignalLow, SignalMedium, Tag, UserCircle, X, ListFilter, Calendar as CalendarIcon, CalendarPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem } from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { useState, useEffect } from "react";
import { nanoid } from "nanoid";

export enum FilterType {
  STATUS = "status",
  ASSIGNEE = "assignee",
  LABELS = "labels",
  PRIORITY = "priority",
  DUE_DATE = "dueDate"
}

export enum FilterOperator {
  IS = "is",
  IS_NOT = "isNot",
  CONTAINS = "contains",
  DOES_NOT_CONTAIN = "doesNotContain",
  BEFORE = "before",
  AFTER = "after"
}

export enum Status {
  BACKLOG = "backlog",
  TODO = "todo",
  IN_PROGRESS = "inProgress",
  DONE = "done",
  CANCELED = "canceled"
}

export enum Assignee {
  NONE = "none",
  ME = "me",
  BOB = "bob",
  ALICE = "alice"
}

export enum Labels {
  BUG = "bug",
  FEATURE = "feature",
  IMPROVEMENT = "improvement",
  DOCUMENTATION = "documentation"
}

export enum Priority {
  NONE = "none",
  LOW = "low",
  MEDIUM = "medium",
  HIGH = "high"
}

export enum DueDate {
  NONE = "none",
  TODAY = "today",
  TOMORROW = "tomorrow",
  NEXT_WEEK = "nextWeek"
}

export const filterViewOptions = [
  {
    value: FilterType.STATUS,
    label: "Status",
    icon: AlertCircle
  },
  {
    value: FilterType.ASSIGNEE,
    label: "Assignee",
    icon: UserCircle
  },
  {
    value: FilterType.LABELS,
    label: "Labels",
    icon: Tag
  },
  {
    value: FilterType.PRIORITY,
    label: "Priority",
    icon: SignalHigh
  },
  {
    value: FilterType.DUE_DATE,
    label: "Due Date",
    icon: CalendarIcon
  }
];

export const statusFilterOptions = [
  {
    value: Status.BACKLOG,
    label: "Backlog",
    icon: Circle
  },
  {
    value: Status.TODO,
    label: "Todo",
    icon: CircleDashed
  },
  {
    value: Status.IN_PROGRESS,
    label: "In Progress",
    icon: CircleDotDashed
  },
  {
    value: Status.DONE,
    label: "Done",
    icon: CheckCircle
  },
  {
    value: Status.CANCELED,
    label: "Canceled",
    icon: XCircle
  }
];

export const assigneeFilterOptions = [
  {
    value: Assignee.NONE,
    label: "Unassigned",
    icon: CircleEllipsis
  },
  {
    value: Assignee.ME,
    label: "Me",
    icon: UserCircle
  },
  {
    value: Assignee.BOB,
    label: "Bob",
    icon: UserCircle
  },
  {
    value: Assignee.ALICE,
    label: "Alice",
    icon: UserCircle
  }
];

export const labelFilterOptions = [
  {
    value: Labels.BUG,
    label: "Bug",
    icon: Tag
  },
  {
    value: Labels.FEATURE,
    label: "Feature",
    icon: Tag
  },
  {
    value: Labels.IMPROVEMENT,
    label: "Improvement",
    icon: Tag
  },
  {
    value: Labels.DOCUMENTATION,
    label: "Documentation",
    icon: Tag
  }
];

export const priorityFilterOptions = [
  {
    value: Priority.NONE,
    label: "None",
    icon: SignalLow
  },
  {
    value: Priority.LOW,
    label: "Low",
    icon: SignalLow
  },
  {
    value: Priority.MEDIUM,
    label: "Medium",
    icon: SignalMedium
  },
  {
    value: Priority.HIGH,
    label: "High",
    icon: SignalHigh
  }
];

export const dateFilterOptions = [
  {
    value: DueDate.NONE,
    label: "No due date",
    icon: CalendarIcon
  },
  {
    value: DueDate.TODAY,
    label: "Today",
    icon: CalendarIcon
  },
  {
    value: DueDate.TOMORROW,
    label: "Tomorrow",
    icon: CalendarPlus
  },
  {
    value: DueDate.NEXT_WEEK,
    label: "Next week",
    icon: CalendarPlus
  }
];

export const filterViewToFilterOptions = {
  [FilterType.STATUS]: statusFilterOptions,
  [FilterType.ASSIGNEE]: assigneeFilterOptions,
  [FilterType.LABELS]: labelFilterOptions,
  [FilterType.PRIORITY]: priorityFilterOptions,
  [FilterType.DUE_DATE]: dateFilterOptions
};

export interface Filter {
  id: string;
  type: FilterType;
  operator: FilterOperator;
  value: string[];
}

interface FiltersProps {
  filters: Filter[];
  onChange: (filters: Filter[]) => void;
}

export function AnimateChangeInHeight({ children }: { children: React.ReactNode }) {
  return (
    <div className="transition-[height] duration-200 ease-in-out overflow-hidden">
      {children}
    </div>
  );
}

export function ComboboxDemo() {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState("");

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-[200px] justify-between"
        >
          {value
            ? filterViewOptions.find((option) => option.value === value)?.label
            : "Select filter..."}
          <ListFilter className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[200px] p-0">
        <Command>
          <CommandInput placeholder="Search filter..." />
          <CommandEmpty>No filter found.</CommandEmpty>
          <CommandGroup>
            {filterViewOptions.map((option) => (
              <CommandItem
                key={option.value}
                value={option.value}
                onSelect={(currentValue) => {
                  setValue(currentValue === value ? "" : currentValue);
                  setOpen(false);
                }}
              >
                <Check
                  className={cn(
                    "mr-2 h-4 w-4",
                    value === option.value ? "opacity-100" : "opacity-0"
                  )}
                />
                {option.label}
              </CommandItem>
            ))}
          </CommandGroup>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

export function Filters({ filters, onChange }: FiltersProps) {
  const addFilter = () => {
    const newFilter: Filter = {
      id: nanoid(),
      type: FilterType.STATUS,
      operator: FilterOperator.IS,
      value: []
    };
    onChange([...filters, newFilter]);
  };

  const removeFilter = (id: string) => {
    onChange(filters.filter((f) => f.id !== id));
  };

  const updateFilter = (id: string, updates: Partial<Filter>) => {
    onChange(
      filters.map((f) => (f.id === id ? { ...f, ...updates } : f))
    );
  };

  return (
    <div className="flex flex-col gap-2">
      {filters.map((filter) => (
        <div key={filter.id} className="flex items-center gap-2">
          <ComboboxDemo />
          <Button
            variant="ghost"
            size="icon"
            onClick={() => removeFilter(filter.id)}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      ))}
      <Button onClick={addFilter} variant="outline" className="w-fit">
        Add filter
      </Button>
    </div>
  );
}